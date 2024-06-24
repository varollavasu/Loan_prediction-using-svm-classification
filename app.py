from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the model
model = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Collect data from form
        gender = request.form['Gender']
        married = request.form['Married']
        dependents = float(request.form['Dependents'])
        education = request.form['Education']
        self_employed = request.form['Self_Employed']
        applicant_income = float(request.form['ApplicantIncome'])
        coapplicant_income = float(request.form['CoapplicantIncome'])
        loan_amount = float(request.form['LoanAmount'])
        loan_amount_term = float(request.form['Loan_Amount_Term'])
        credit_history = float(request.form['Credit_History'])
        property_area = request.form['Property_Area']

        # Convert categorical inputs to numerical
        gender = 1 if gender == 'Male' else 0
        married = 1 if married == 'Yes' else 0
        self_employed = 1 if self_employed == 'Yes' else 0
        education = 1 if education == 'Graduate' else 0
        property_area_mapping = {'Rural': 0, 'Semiurban': 1, 'Urban': 2}
        property_area = property_area_mapping[property_area]

        # Create numpy array from form data
        input_data = np.array([[gender, married, dependents, education, self_employed, 
                                applicant_income, coapplicant_income, loan_amount, 
                                loan_amount_term, credit_history, property_area]])

        # Make prediction
        prediction = model.predict(input_data)

        # Return the result to the 'result.html' template
        return render_template("result.html", prediction=prediction[0])

    except Exception as e:
        print(f"Error occurred: {e}")
        return render_template("home.html", error_message="An error occurred during prediction. Please check your input values.")

if __name__ == "__main__":
    app.run(debug=True)
