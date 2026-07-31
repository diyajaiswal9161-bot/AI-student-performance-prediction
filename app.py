from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    hours = float(request.form["hours"])
    attendance = float(request.form["attendance"])
    previous = float(request.form["previous"])
    assignments = float(request.form["assignments"])

    prediction = model.predict([[hours, attendance, previous, assignments]])

    score = round(prediction[0], 2)

    if score >= 85:
        performance = "Excellent"
    elif score >= 70:
        performance = "Good"
    elif score >= 50:
        performance = "Average"
    else:
        performance = "Needs Improvement"

    return render_template(
        "result.html",
        score=score,
        performance=performance
    )

if __name__ == "__main__":
    app.run(debug=True)