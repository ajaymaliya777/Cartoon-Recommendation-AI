from flask import Flask, request, jsonify
import joblib
import numpy as np

# Flask app create
app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")


@app.route("/")
def home():
    return "Cartoon Recommendation API is Running 🚀"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    age = data["Age"]
    action = data["Likes_Action"]
    comedy = data["Likes_Comedy"]
    superhero = data["Likes_Superhero"]

    features = np.array([[age, action, comedy, superhero]])

    prediction = model.predict(features)

    return jsonify({
        "Recommended_Show": prediction[0]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
