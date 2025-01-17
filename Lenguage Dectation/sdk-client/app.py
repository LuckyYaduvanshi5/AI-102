from flask import Flask, render_template, request
from dotenv import load_dotenv
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

app = Flask(__name__)

# Load environment variables
load_dotenv()
ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
ai_key = os.getenv('AI_SERVICE_KEY')

# Initialize Azure Text Analytics Client
credential = AzureKeyCredential(ai_key)
client = TextAnalyticsClient(endpoint=ai_endpoint, credential=credential)

def get_language(text):
    try:
        # Call the Azure service to detect the language
        detected_language = client.detect_language(documents=[text])[0]
        return detected_language.primary_language.name
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    user_text = request.form['user_text']
    if user_text:
        language = get_language(user_text)
    else:
        language = "No text entered."
    return render_template('result.html', language=language)

if __name__ == '__main__':
    app.run(debug=True)
