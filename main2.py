from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import json

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, AIMessage
from langchain_groq.chat_models import ChatGroq

# Import your doctor database
import data

load_dotenv()
app = Flask(__name__)

# Enable CORS for all origins and methods
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type"])

# Set your Groq API key
groq_api_key = os.getenv("groq_api_key") or "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"

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

# Serialize doctor database as JSON for context
doctor_context = json.dumps(data.doctors_db, indent=2)

# Build a ChatPromptTemplate that includes:
# 1) System message with doctor DB
# 2) System message with instructions
# 3) User message placeholder
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=(
        "You are an expert medical professional with extensive knowledge of diseases, symptoms, and treatments. "
        "You have access to the following doctor database (do not expose the raw list unless specifically asked):\n" + doctor_context
    )),
    SystemMessage(content=(
        "When a user describes symptoms, use the doctor database to suggest appropriate specialists by name, hospital, and location. "
        "Also analyze the symptoms and provide possible conditions, recommended treatments, and self-care advice. "
        "If symptoms suggest an emergency, advise the user to seek immediate professional help. "
        "Always respond clearly and concisely in English."
    )),
    ("user", "{query}")
])

@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    try:
        payload = request.get_json()
        if not payload:
            return jsonify({"error": "No JSON data received"}), 400

        query = payload.get("message", "").strip()
        if not query:
            return jsonify({"error": "Empty message"}), 400

        print(f"Received query: {query}")

        # Combine prompt and LLM
        chain = prompt | llm
        response = chain.invoke({"query": query})

        result = response.content if isinstance(response, AIMessage) else str(response)
        print(f"Generated response: {result[:100]}...")

        return jsonify({"reply": result})

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": f"Server error: {e}"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "message": "Backend is running"})

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
