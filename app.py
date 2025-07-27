import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import joblib
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
from sklearn.decomposition import PCA

st.set_page_config(layout="wide")
st.title("Streamlit AutoML with Full EDA and ML Training")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())

    page = st.sidebar.radio("Select Section", ["EDA", "Model Training"])

    # ============================== EDA SECTION ============================== #
    if page == "EDA":
        st.header("Exploratory Data Analysis (EDA)")

        selected_vars = st.sidebar.multiselect("Select variable(s) for EDA", df.columns)

        eda_type = st.sidebar.selectbox("Choose EDA Type", [
            "Descriptive Statistics",
            "Histogram",
            "Box Plot",
            "Correlation Heatmap",
            "Pairplot (max 5 vars)",
            "Countplot (Categorical)",
            "Outlier Detection (IQR)"
        ])

        if not selected_vars:
            st.info("Please select at least one variable to begin EDA.")
        else:
            st.subheader(f"EDA: {eda_type}")

            if eda_type == "Descriptive Statistics":
                st.write(df[selected_vars].describe(include='all'))

            elif eda_type == "Histogram":
                for col in selected_vars:
                    if np.issubdtype(df[col].dtype, np.number):
                        fig, ax = plt.subplots()
                        sns.histplot(df[col], kde=True, ax=ax)
                        ax.set_title(f"Histogram of {col}")
                        st.pyplot(fig)

            elif eda_type == "Box Plot":
                for col in selected_vars:
                    if np.issubdtype(df[col].dtype, np.number):
                        fig, ax = plt.subplots()
                        sns.boxplot(y=df[col], ax=ax)
                        ax.set_title(f"Boxplot of {col}")
                        st.pyplot(fig)

            elif eda_type == "Correlation Heatmap":
                numeric_df = df[selected_vars].select_dtypes(include=[np.number])
                if numeric_df.shape[1] >= 2:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
                    st.pyplot(fig)
                else:
                    st.warning("Need at least 2 numeric variables for correlation heatmap.")

            elif eda_type == "Pairplot (max 5 vars)":
                if len(selected_vars) <= 5:
                    fig = sns.pairplot(df[selected_vars].dropna())
                    st.pyplot(fig)
                else:
                    st.warning("Select 5 or fewer variables for pairplot.")

            elif eda_type == "Countplot (Categorical)":
                for col in selected_vars:
                    if df[col].dtype == 'object' or df[col].nunique() < 20:
                        fig, ax = plt.subplots()
                        sns.countplot(x=df[col], order=df[col].value_counts().index, ax=ax)
                        ax.set_title(f"Countplot of {col}")
                        ax.tick_params(axis='x', rotation=45)
                        st.pyplot(fig)

            elif eda_type == "Outlier Detection (IQR)":
                for col in selected_vars:
                    if np.issubdtype(df[col].dtype, np.number):
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower = Q1 - 1.5 * IQR
                        upper = Q3 + 1.5 * IQR
                        outliers = df[(df[col] < lower) | (df[col] > upper)]
                        st.write(f"Outliers in **{col}**: {len(outliers)}")
                        st.dataframe(outliers)

    # ============================== MODEL TRAINING SECTION ============================== #
    elif page == "Model Training":
        st.header("Model Training and Statistical Summaries")
        y_col = st.selectbox("Select Target Variable (Y)", df.columns)
        x_cols = st.multiselect("Select Feature Variables (X)", [col for col in df.columns if col != y_col])
        normalize = st.sidebar.checkbox("Normalize numeric features", value=True)

        if x_cols:
            model_choice = st.selectbox("Select Model", [
                "Linear Regression", "Polynomial Regression", "Ridge Regression", "Lasso Regression",
                "Logistic Regression", "Naive Bayes (Gaussian)", "K-Nearest Neighbors", "Support Vector Machine",
                "Decision Tree", "Random Forest", "PCA (Unsupervised)"
            ])
            params = {}
            if model_choice == "K-Nearest Neighbors":
                params['n_neighbors'] = st.sidebar.slider("n_neighbors", 1, 15, 5)
            elif model_choice == "Support Vector Machine":
                params['C'] = st.sidebar.slider("C", 0.01, 10.0, 1.0)
            elif model_choice in ["Decision Tree", "Random Forest"]:
                params['max_depth'] = st.sidebar.slider("max_depth", 1, 20, 5)
            if model_choice == "Random Forest":
                params['n_estimators'] = st.sidebar.slider("n_estimators", 10, 200, 100)

            def show_feature_importance(model, feature_names, title):
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                elif hasattr(model, "coef_"):
                    importances = model.coef_.flatten()
                else:
                    return
                fig, ax = plt.subplots()
                sorted_idx = np.argsort(importances)[::-1]
                sorted_names = np.array(feature_names)[sorted_idx]
                sorted_importances = importances[sorted_idx]
                ax.barh(sorted_names, sorted_importances)
                ax.set_xlabel("Importance")
                ax.set_title(f"{title} - Feature Importance")
                st.pyplot(fig)

            if st.button("Train Model"):
                X = df[x_cols]
                y = df[y_col]
                if y.dtype == 'object':
                    y = pd.factorize(y)[0]
                X = pd.get_dummies(X)
                feature_names = X.columns
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                if normalize:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)

                model = None
                if model_choice == "Linear Regression":
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    st.write("R² Score:", r2_score(y_test, y_pred))
                    fig, ax = plt.subplots()
                    ax.scatter(y_pred, y_test - y_pred)
                    ax.axhline(0, color='red', linestyle='--')
                    ax.set_title("Residual Plot")
                    st.pyplot(fig)
                    X_sm = sm.add_constant(X)
                    sm_model = sm.OLS(y, X_sm).fit()
                    st.text("Statistical Summary:")
                    st.text(sm_model.summary())

                elif model_choice == "Polynomial Regression":
                    poly = PolynomialFeatures(degree=2)
                    X_train = poly.fit_transform(X_train)
                    X_test = poly.transform(X_test)
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    st.write("R² Score:", r2_score(y_test, y_pred))

                elif model_choice == "Ridge Regression":
                    model = Ridge()
                elif model_choice == "Lasso Regression":
                    model = Lasso()
                elif model_choice == "Logistic Regression":
                    model = LogisticRegression()
                elif model_choice == "Naive Bayes (Gaussian)":
                    model = GaussianNB()
                elif model_choice == "K-Nearest Neighbors":
                    model = KNeighborsClassifier(n_neighbors=params['n_neighbors'])
                elif model_choice == "Support Vector Machine":
                    model = SVC(C=params['C'], probability=True)
                elif model_choice == "Decision Tree":
                    model = DecisionTreeClassifier(max_depth=params['max_depth'])
                elif model_choice == "Random Forest":
                    model = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'])

                if model_choice not in ["PCA (Unsupervised)", "Polynomial Regression"]:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    if model_choice in ["Linear Regression", "Ridge Regression", "Lasso Regression"]:
                        st.write("R² Score:", r2_score(y_test, y_pred))
                    else:
                        st.write("Accuracy:", accuracy_score(y_test, y_pred))
                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots()
                        ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)
                        st.pyplot(fig)
                    if model_choice == "Logistic Regression":
                        X_sm = sm.add_constant(X)
                        sm_model = sm.Logit(y, X_sm).fit()
                        st.text("Statistical Summary:")
                        st.text(sm_model.summary())

                if model_choice == "Decision Tree":
                    fig, ax = plt.subplots(figsize=(10, 6))
                    plot_tree(model, filled=True, feature_names=feature_names)
                    st.pyplot(fig)

                if model_choice == "PCA (Unsupervised)":
                    numeric_df = df[x_cols].select_dtypes(include=[np.number])
                    X_scaled = StandardScaler().fit_transform(numeric_df)
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled)
                    st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)
                    fig, ax = plt.subplots()
                    ax.scatter(X_pca[:, 0], X_pca[:, 1])
                    ax.set_xlabel("PC1")
                    ax.set_ylabel("PC2")
                    ax.set_title("PCA - First Two Principal Components")
                    st.pyplot(fig)

                if model_choice not in ["PCA (Unsupervised)", "Polynomial Regression"]:
                    show_feature_importance(model, feature_names, model_choice)
