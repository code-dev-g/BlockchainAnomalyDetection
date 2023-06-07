import streamlit as st
import pickle

st.title("Anomaly Detection in Blockchain Networks")

with open('../MachineLearning/Models/adaboost_model.sav', 'rb') as f:
    ada_boost_model = pickle.load(f)

with open('../MachineLearning/Models/decision_tree_model.sav', 'rb') as f:
    decision_tree_model = pickle.load(f)

with open('../MachineLearning/Models/gradient_boosting_model.sav', 'rb') as f:
    gradient_boosting_model = pickle.load(f)

with open('../MachineLearning/Models/k_nearest_neighbors_model.sav', 'rb') as f:
    k_nearest_neighbors_model = pickle.load(f)

with open('../MachineLearning/Models/logistic_regression_model.sav', 'rb') as f:
    logistic_regression_model = pickle.load(f)

with open('../MachineLearning/Models/naive_bayes_model.sav', 'rb') as f:
    naive_bayes_model = pickle.load(f)

with open('../MachineLearning/Models/random_forest_model.sav', 'rb') as f:
    random_forest_model = pickle.load(f)

with open('../MachineLearning/Models/support_vector_machine_model.sav', 'rb') as f:
    support_vector_machine_model = pickle.load(f)

with open('../MachineLearning/Models/xgboost_model.sav', 'rb') as f:
    xgboost_model = pickle.load(f)

# give a drop down to choose the model
model = st.selectbox("Choose the model", ["Ada Boost", "Decision Tree", "Gradient Boosting", "K Nearest Neighbors",
                                            "Logistic Regression", "Naive Bayes", "Random Forest", "Support Vector Machine",
                                            "XGBoost"])

# give a set of fields to enter the values
# input fields
# 1. Indegree
# 2. Outdegree
# 3. IN BTC
# 4. OUT BTC

indegree = st.number_input("Enter the indegree", step=1)
outdegree = st.number_input("Enter the outdegree", step=1)
in_btc = st.number_input("Enter the in btc", step=10)
out_btc = st.number_input("Enter the out btc", step=10)

is_in_malicious = st.checkbox("Is the input node malicious?")
is_out_malicious = st.checkbox("Is the output node malicious?")

is_in_malicious = 1 if is_in_malicious else 0
is_out_malicious = 1 if is_out_malicious else 0

input_data = [[indegree, outdegree, in_btc, out_btc, is_in_malicious, is_out_malicious]]

prediction = None

# predict the value
if st.button("Predict"):
    if model == "Ada Boost":
        prediction = ada_boost_model.predict(input_data)
    elif model == "Decision Tree":
        prediction = decision_tree_model.predict(input_data)
    elif model == "Gradient Boosting":
        prediction = gradient_boosting_model.predict(input_data)
    elif model == "K Nearest Neighbors":
        prediction = k_nearest_neighbors_model.predict(input_data)
    elif model == "Logistic Regression":
        prediction = logistic_regression_model.predict(input_data)
    elif model == "Naive Bayes":
        prediction = naive_bayes_model.predict(input_data)
    elif model == "Random Forest":
        prediction = random_forest_model.predict(input_data)
    elif model == "Support Vector Machine":
        prediction = support_vector_machine_model.predict(input_data)
    elif model == "XGBoost":
        prediction = xgboost_model.predict(input_data)

    if prediction[0] == 1:
        st.header("The transaction is anomalous")
    else:
        st.header("The transaction is not anomalous")



