from flask import Flask, request, jsonify
import openai
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

openai.api_key = "your_api_key"

@app.route('/')
def home():
    return "Flask app is running"

@app.route('/classify', methods=['POST'])
def classify_message():
    data = request.json
    message = data.get('message', '')

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system", "content":"You are a classifier. Classify the following message as 'Hate Speech' or 'Non-Hate Speech'."},
            {"role":"user", "content":message}
        ]
    )
    classification = response.choices[0].message['content']

    return jsonify({"message": message, "classification": classification})


if __name__ == "__main__":
    app.run(debug=True)



