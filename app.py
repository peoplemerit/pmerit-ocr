from flask import Flask, request, render_template, jsonify
from PIL import Image
import pytesseract
#import tf_keras
import tensorflow as tf
from transformers import pipeline

app = Flask(__name__)

# Load NLP pipelines
#sentiment_analysis = pipeline("sentiment-analysis")
#summarization = pipeline("summarization")

# Load NLP pipelines with explicit model names and revisions
sentiment_analysis = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", revision="714eb0f")
summarization = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", revision="a4f8f3e")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")  # Create an HTML form to accept images and text

# Endpoint to handle image uploads
@app.route("/upload-image", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return "No file part"
    
    file = request.files["image"]
    if file.filename == "":
        return "No selected file"
    
    image = Image.open(file)
    extracted_text = pytesseract.image_to_string(image)
    
    return jsonify({"extracted_text": extracted_text})

# New Endpoint to handle text input for NLP
@app.route("/analyze-text", methods=["POST"])
def analyze_text():
    data = request.get_json()
    text = data.get("text", "")

    if text.strip() == "":
        return jsonify({"error": "No text provided"}), 400

    # Perform NLP tasks
    sentiment = sentiment_analysis(text)
    summary = summarization(text, max_length=50, min_length=25, do_sample=False)

    return jsonify({
        "sentiment": sentiment,
        "summary": summary[0]["summary_text"]
    })

# Run the app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
