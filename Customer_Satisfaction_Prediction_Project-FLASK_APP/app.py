from flask import Flask, request, render_template
import joblib
import pandas as pd
import xgboost as xgb

model = xgb.XGBClassifier()

# Load model and tools
model.load_model('customer_satisfaction_model_xgb.json')
scaler = joblib.load('scaler.pkl')
training_columns = joblib.load('training_columns.pkl')

# Flask app
app = Flask(__name__)

@app.route('/')
def form():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs 
        age = int(request.form['age'])
        product = int(request.form['product'])
        response_time = float(request.form['response_time'])
        sentiment = float(request.form['sentiment'])
        tickets = int(request.form['tickets'])
        months = int(request.form['months'])
        gender = request.form['gender']
        ticket_type = request.form['ticket_type']
        age_group = request.form['age_group']

        # Build feature dictionary
        input_dict = {
            'Customer Age' : age,
            'Response Time Delta': response_time,
            'Sentiment': sentiment,
            'Tickets per Customer': tickets,
            'Months Since Start': months,
            f'Customer Gender_{gender}': 1,
            f'Ticket Type_{ticket_type}': 1,
            f'Age_Group_{age_group}': 1

        }

        # Add missing columns
        full_input = {}
        for col in training_columns:
            if col in input_dict:
              full_input[col] = input_dict[col]
            else:
              full_input[col] = 0

        # Creating DataFrame and Numerical Scaling  
        input_df = pd.DataFrame([full_input])
        numerical_cols = ['Customer Age', 'Response Time Delta', 'Sentiment', 'Tickets per Customer', 'Months Since Start']
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
 

        # Predict
        prediction = int(model.predict(input_df)[0])
        return render_template('index.html', prediction=prediction)

    except Exception as e:
        return render_template('index.html', prediction=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)


