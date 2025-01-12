from flask import Flask, request, jsonify
import openai

app = Flask(__name__)
openai.api_key = "your_openai_api_key"  # replace with your actual API key

def get_chatgpt_response(user_input):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use "gpt-4" if you have access
        messages=[{"role": "user", "content": user_input}]
    )
    return response['choices'][0]['message']['content']

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    response = get_chatgpt_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
