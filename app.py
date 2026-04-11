from flask import Flask, request, jsonify, render_template
from transformers import pipeline

app = Flask(__name__)

# load BERT instead of LSTM
classifier = pipeline("sentiment-analysis")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    review = data["review"]
    result = classifier(review)[0]

    sentiment = result["label"].capitalize()
    confidence = f"{result['score']:.0%}"

    return jsonify({"sentiment": sentiment, "confidence": confidence})


if __name__ == "__main__":
    app.run(debug=True)