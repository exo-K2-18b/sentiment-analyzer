from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

app = Flask(__name__)

model = tf.keras.models.load_model("model/sentiment_analyzer.keras")
word_index = tf.keras.datasets.imdb.get_word_index()

def predict_sentiment(review):
    review = review.lower()
    review = re.sub(r"[^a-z\s]", "", review)
    words = review.split()
    sequence = [min(word_index.get(word, 2), 9999) for word in words]
    padded = pad_sequences([sequence], maxlen=200)
    prediction = model.predict(padded, verbose=0)[0][0]
    return float(prediction)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    review = data["review"]
    prediction = predict_sentiment(review)
    if prediction > 0.5:
        sentiment = "Positive"
        confidence = f"{prediction:.0%}"
    else:
        sentiment = "Negative"
        confidence = f"{1-prediction:.0%}"
    return jsonify({"sentiment": sentiment, "confidence": confidence})

import os

if __name__ == "__main__":
    # Railway provides a port, or we use 8080 as a backup
    port = int(os.environ.get("PORT", 8080))
    # '0.0.0.0' tells Flask to listen to all public requests
    app.run(host='0.0.0.0', port=port)