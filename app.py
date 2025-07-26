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
st.title("AutoML GUI – Streamlit App with Statistical Summaries")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())

    y_col = st.selectbox("Select Target Variable (Y)", df.columns)
    x_cols = st.multiselect("Select Feature Variables (X)", [col for col in df.columns if col != y_col])

    normalize = st.sidebar.checkbox("Normalize numeric features", value=True)

    if x_cols:
        model_choice = st.selectbox("Select Model", [
            "Linear Regression",
            "Polynomial Regression",
            "Ridge Regression",
            "Lasso Regression",
            "Logistic Regression",
            "Naive Bayes (Gaussian)",
            "K-Nearest Neighbors",
            "Support Vector Machine",
            "Decision Tree",
            "Random Forest",
            "PCA (Unsupervised)"
        ])

        # Sidebar hyperparameter sliders
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
            metrics_text = ""

            if model_choice == "Linear Regression":
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                st.write("R² Score:", r2)
                fig, ax = plt.subplots()
                ax.scatter(y_pred, y_test - y_pred)
                ax.axhline(0, color='red', linestyle='--')
                ax.set_title("Residual Plot")
                st.pyplot(fig)

                # Statsmodels summary
                X_sm = sm.add_constant(X)
                sm_model = sm.OLS(y, X_sm).fit()
                st.text("Statistical Summary:")
                st.text(sm_model.summary())

            elif model_choice == "Polynomial Regression":
                poly = PolynomialFeatures(degree=2)
                X_train_poly = poly.fit_transform(X_train)
                X_test_poly = poly.transform(X_test)
                model = LinearRegression()
                model.fit(X_train_poly, y_train)
                y_pred = model.predict(X_test_poly)
                r2 = r2_score(y_test, y_pred)
                st.write("R² Score:", r2)

            elif model_choice == "Ridge Regression":
                model = Ridge()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.write("R² Score:", r2_score(y_test, y_pred))

            elif model_choice == "Lasso Regression":
                model = Lasso()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.write("R² Score:", r2_score(y_test, y_pred))

            elif model_choice == "Logistic Regression":
                model = LogisticRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.write("Accuracy:", accuracy_score(y_test, y_pred))
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)
                st.pyplot(fig)

                # Statsmodels logistic regression
                X_sm = sm.add_constant(X)
                sm_model = sm.Logit(y, X_sm).fit()
                st.text("Statistical Summary:")
                st.text(sm_model.summary())

            elif model_choice == "Naive Bayes (Gaussian)":
                model = GaussianNB()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.write("Accuracy:", accuracy_score(y_test, y_pred))

            elif model_choice == "K-Nearest Neighbors":
                model = KNeighborsClassifier(n_neighbors=params['n_neighbors'])
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.write("Accuracy:", accuracy_score(y_test, y_pred))

            elif model_choice == "Support Vector Machine":
                model = SVC(C=params['C'], probability=True)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.write("Accuracy:", accuracy_score(y_test, y_pred))

            elif model_choice == "Decision Tree":
                model = DecisionTreeClassifier(max_depth=params['max_depth'])
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.write("Accuracy:", accuracy_score(y_test, y_pred))
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_tree(model, filled=True, feature_names=feature_names)
                st.pyplot(fig)

            elif model_choice == "Random Forest":
                model = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'])
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.write("Accuracy:", accuracy_score(y_test, y_pred))

            elif model_choice == "PCA (Unsupervised)":
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
                buf = io.BytesIO()
                joblib.dump(model, buf)
                st.download_button("Download Trained Model", buf.getvalue(), file_name="trained_model.pkl")
