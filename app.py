from flask import Flask, request, render_template
import numpy as np
import pickle
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize encoders and scaler
# Replace with actual categories and fitting data used during training
oe_geography = OrdinalEncoder(categories=[['France', 'Spain', 'Germany']])
oe_gender = OrdinalEncoder(categories=[['Male', 'Female']])
std = StandardScaler()

# Dummy fit for scaler - replace with actual data
std.fit([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])

@app.route("/", methods=['GET', 'POST'])
def predictions():
    if request.method == 'POST':
        form_data = request.form

        try:
            # Extract numeric fields
            numeric_fields = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
            numeric_values = [float(form_data.get(field, 0)) for field in numeric_fields]

            # Extract and encode string fields
            geography = form_data.get('Geography')
            gender = form_data.get('Gender')

            if geography:
                geography_encoded = oe_geography.transform([[geography]])
                numeric_values.append(geography_encoded[0][0])
            else:
                numeric_values.append(0)

            if gender:
                gender_encoded = oe_gender.transform([[gender]])
                numeric_values.append(gender_encoded[0][0])
            else:
                numeric_values.append(0)

            # Combine and standardize features
            features = np.array(numeric_values).reshape(1, -1)
            features_scaled = std.transform(features)

            # Make prediction
            prediction = model.predict(features_scaled)
            prediction_result = prediction[0]

        except Exception as e:
            return render_template('index.html', prediction=f"Error: {str(e)}")

        return render_template('index.html', prediction=prediction_result)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
