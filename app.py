from flask import Flask, render_template, request
import numpy as np
import joblib  # or pickle, depending on your model

app = Flask(__name__)

# Load the model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        temperature = float(request.form["Temperature"])
        humidity = float(request.form["Humidity"])
        wind_speed = float(request.form["WindSpeed"])
        precipitation = float(request.form["Precipitation"])
        cloud_cover = float(request.form["CloudCover"])
        pressure = float(request.form["Pressure"])

        features = np.array([[temperature, humidity, wind_speed, precipitation, cloud_cover, pressure]])
        prediction = model.predict(features)[0]

        result = "Yes 🌧️" if prediction == 1 else "No ☀️"

        return render_template("result.html", result=result)

    except Exception as e:
        return render_template("result.html", result=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
