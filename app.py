import pickle
import pandas as pd
from flask import Flask, request

app = Flask(__name__)

model_pickle = open("./artefacts/classifier.pkl", "rb")
clf = pickle.load(model_pickle)


@app.route("/ping", methods=['GET'])
def ping():
    return {"message": "Hi there, I'm working!!"}


@app.route("/params", methods=['GET'])
def get_application_params():
    """
    """
    parameters = {
    'City': "<C1/C2>", 'Education_Level': "<1/2>", 'Income': "<Income Amount>", 'Age': "<Age>",
    'Gender': "<1/0>", 'Joining Designation': "<1/2>",' Grade': "<1/2>",
    'Total_Business_Value': "<Total Business Value>", 'Quarterly_Rating': "<Quarterly Rating>",
    'Inc_Flag': "<1/0>", 'QR_Flag': "<1/0>", 'Duration': "<Duration>"
    }

    return parameters


##defining the endpoint which will make the prediction
@app.route("/predict", methods=['POST'])
def prediction():
    """
    Returns Ola Churn status using ML model.
    """

    churn_req = request.get_json()    

    city = churn_req["City"]
    education = churn_req["Education_Level"]
    income = churn_req["Income"]
    age = churn_req["Age"]
    gender = churn_req["Gender"]
    join_dsgn = churn_req["Joining Designation"]
    grade = churn_req["Grade"]
    business_val = churn_req["Total Business Value"]
    qtr_rat = churn_req["Quarterly Rating"]
    inc_flag = churn_req["Inc_Flag"] 
    qr_flag = churn_req["QR_Flag"] 
    duration = churn_req["Duration"]

    data = [[city, education, income, age, gender, join_dsgn,
                    grade, business_val, qtr_rat, inc_flag, qr_flag, duration]]

    columns = ['City', 'Education_Level', 'Income', 'Age', 'Gender','Joining Designation', 'Grade', 
                'Total Business Value','Quarterly Rating', 'Inc_Flag', 'QR_Flag', 'Duration']

    query_data = pd.DataFrame(data=data, columns=columns)

    result = clf.predict(query_data)

    if result == 1:
        pred = "The driver is going to Churn."
    else:
        pred = "The driver is not going to Churn."
    
    return {"ola_driver_churn_status": pred}
