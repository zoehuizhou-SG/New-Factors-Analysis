import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Multi-Factor Analysis Tool")

# File upload or sample data
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
else:
    st.info("No file uploaded. Using sample data from creditcarddata.csv")
    df = pd.read_csv("creditcarddata.csv")

with st.expander("Data Preview", expanded=True):
    st.write("Data Preview:")
    st.dataframe(df.head())

# Select numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Exclude irrelevant columns
exclude_cols = ['customer_id', 'credit_card_number']
numerical_cols = [col for col in numerical_cols if col not in exclude_cols]

if len(numerical_cols) == 0:
    st.error("No numerical columns found in the data after excluding irrelevant ones. Please upload a CSV with numerical data.")
else:
    st.write(f"Numerical columns for analysis: {numerical_cols}")

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    corr = df[numerical_cols].corr()
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax_corr, center=0, fmt='.2f')
    st.pyplot(fig_corr)

    # Number of factors
    n_factors = st.slider("Select number of factors", min_value=1, max_value=min(len(numerical_cols), 10), value=2)

    # Perform Factor Analysis
    if st.button("Run Factor Analysis"):
        fa = FactorAnalysis(n_components=n_factors, random_state=42)
        fa.fit(df[numerical_cols])

        # Factor loadings
        loadings = pd.DataFrame(
            fa.components_.T,
            index=numerical_cols,
            columns=[f'Factor {i+1}' for i in range(n_factors)]
        )

        st.subheader("Factor Loadings")
        st.dataframe(loadings)

        # Notes explaining each factor's likely meaning
        st.subheader("Factor Meaning Notes")
        factor_explanations = []
        for i in range(n_factors):
            factor = f'Factor {i+1}'
            top_vars = loadings[factor].abs().sort_values(ascending=False).head(4).index.tolist()
            labels = []
            if any(v in top_vars for v in ['credit_score', 'annual_income', 'credit_limit', 'Credit_History_Years']):
                labels.append('credit strength / capacity')
            if any(v in top_vars for v in ['Debt_to_Income', 'Missed_Payments', 'interest_rate', 'loan_amount']):
                labels.append('repayment risk / stress')
            if 'loan_term' in top_vars:
                labels.append('loan term / maturity')
            if not labels:
                labels.append('mixed financial behavior')
            label_text = ' and '.join(labels)
            explanation = (
                f"{factor} loads most strongly on {', '.join(top_vars)}. "
                f"This factor likely captures {label_text}."
            )
            factor_explanations.append(explanation)
        for note in factor_explanations:
            st.write(f"- {note}")

        # Communalities (variance explained by factors for each variable)
        communalities = np.sum(fa.components_**2, axis=0)
        communality_df = pd.DataFrame(communalities, index=numerical_cols, columns=['Communality'])
        st.subheader("Communalities")
        st.dataframe(communality_df)

        # Heatmap of loadings
        st.subheader("Factor Loadings Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(loadings, annot=True, cmap='coolwarm', ax=ax, center=0)
        st.pyplot(fig)

        # Scatter plot of factors (if at least 2 factors)
        if n_factors >= 2:
            factors = fa.transform(df[numerical_cols])
            st.subheader("Factor Scores Scatter Plot")
            fig2, ax2 = plt.subplots()
            ax2.scatter(factors[:, 0], factors[:, 1], alpha=0.7)
            ax2.set_xlabel('Factor 1')
            ax2.set_ylabel('Factor 2')
            ax2.set_title('Scatter Plot of Factor Scores')
            st.pyplot(fig2)

        # Explained variance approximation (total variance explained)
        total_variance_explained = np.sum(communalities) / len(numerical_cols)
        st.write(f"Approximate total variance explained by {n_factors} factors: {total_variance_explained:.2%}")

        # Credit Rating Assignment
        st.subheader("Credit Rating Assignment")
        st.write("Assigning a credit rating from 1-5 to each customer (5 = perfect credibility) based on key financial variables.")

        # Define weights for variables (positive for good indicators, negative for bad)
        weights = {
            'credit_score': 1.0,
            'annual_income': 0.5,
            'credit_limit': 0.5,
            'Debt_to_Income': -1.0,
            'Missed_Payments': -1.0,
            'Credit_History_Years': 1.0,
            'loan_amount': -0.5,
            'interest_rate': -0.5,
            'loan_term': 0.0,  # neutral
        }

        # Compute composite score
        score = np.zeros(len(df))
        for col in numerical_cols:
            if col in weights and weights[col] != 0:
                w = weights[col]
                # Normalize to 0-1
                col_min, col_max = df[col].min(), df[col].max()
                if col_max > col_min:
                    col_scaled = (df[col] - col_min) / (col_max - col_min)
                else:
                    col_scaled = 0.5  # constant column
                if w < 0:  # invert for bad indicators
                    col_scaled = 1 - col_scaled
                score += w * abs(w) * col_scaled  # weight magnitude

        # Normalize score to 1-5
        score_min, score_max = score.min(), score.max()
        if score_max > score_min:
            rating = 1 + 4 * (score - score_min) / (score_max - score_min)
        else:
            rating = 3.0  # default if all same
        rating = np.clip(rating, 1, 5)
        df_copy = df.copy()
        df_copy['Credit_Rating'] = rating.round(0).astype(int)  # Round to nearest integer 1-5

        st.write("Sample Credit Ratings:")
        st.dataframe(df_copy[['customer_id', 'Credit_Rating']].head(20))

        # Rating distribution
        st.subheader("Credit Rating Distribution")
        fig_rating, ax_rating = plt.subplots()
        ax_rating.hist(df_copy['Credit_Rating'], bins=[1,2,3,4,5,6], edgecolor='black', align='left')
        ax_rating.set_xlabel('Credit Rating (1-5)')
        ax_rating.set_ylabel('Number of Customers')
        ax_rating.set_title('Distribution of Credit Ratings')
        ax_rating.set_xticks([1,2,3,4,5])
        st.pyplot(fig_rating)

        # Sub-portfolio by rating
        st.subheader("View Sub-Portfolio by Credit Rating")
        selected_rating = st.selectbox("Select Credit Rating (1-5)", options=[1,2,3,4,5])
        sub_portfolio = df_copy[df_copy['Credit_Rating'] == selected_rating]
        st.write(f"Customers with Credit Rating {selected_rating}: {len(sub_portfolio)}")
        st.dataframe(sub_portfolio.head(20))  # Show first 20 in sub-portfolio