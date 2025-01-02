from flask import Flask, request, jsonify
import openai
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

openai.api_key = "your-api-key"

@app.route('/')
def home():
    return "Flask app is running"

@app.route('/classify', methods=['POST'])
def classify_message():
    data = request.json
    message = data.get('message', '')

    try:
        classification_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role":"system", "content":"You are a classifier. Classify the following message as 'Hate Speech' or 'Non-Hate Speech'."},
                {"role":"user", "content":message}
            ]
        )
        classification = classification_response.choices[0].message['content']

        gpt_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a friendly chatbot that responds to the user's messages in a polite and engaging manner."},
                {"role": "user", "content":message}       
            ]
        )
        bot_reply = gpt_response.choices[0].message['content']

        if "Hate Speech" in classification:
            bot_reply = "Hate speech detected, " + gpt_response.choices[0].message['content']

        return jsonify({"classification": classification, "reply": bot_reply})
    
    except Exception as e:
        return jsonify({"error": "An error occured", "details": str(e)}), 500
    

if __name__ == "__main__":
    app.run(debug=True)



