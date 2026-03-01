from flask import Flask, render_template, request
import pickle
import pandas as pd


app = Flask(__name__)

model = pickle.load(open("flight_price_model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    airline = request.form['airline']
    source = request.form['source']
    destination = request.form['destination']
    duration = float(request.form['duration'])
    stops = int(request.form['stops'])

    input_dict = {
        'duration': duration,
        'stops': stops
    }

    df = pd.DataFrame([input_dict])

    df = pd.get_dummies(df)

    df = df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(df)

    return render_template(
        "index.html",
        prediction_text=f"Predicted Price: ₹{round(prediction[0],2)}"
    )
if __name__ == "__main__":
    app.run(debug=True)