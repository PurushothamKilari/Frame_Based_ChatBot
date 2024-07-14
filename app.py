from flask import Flask, request, jsonify
from flask_cors import CORS
from dialog_manager import DialogManager

dm = DialogManager("trained_model.joblib")

app = Flask(__name__)
CORS(app)  # To handle CORS issues if your frontend and backend are on different servers

@app.route('/your-endpoint', methods=['POST'])
def handle_message():
    data = request.get_json()
    user_message = data['message']
    
    # Process the user message with your ML model here
    bot_response =dm.process_input(user_message)
    
    return jsonify({'message': bot_response})

# def your_ml_model(message):
#     # This is where you handle the ML model logic
#     return "I'm bot how may I help you."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
