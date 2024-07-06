from flask import Flask, request, jsonify, render_template
import os
import requests
import fitz
import json
import textwrap
import logging
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models
from google.cloud import aiplatform

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

PROJECT_ID="test-vertex-400014"
LOCATION="us-central1"
MODEL_ID="gemini-1.5-pro-001"


def extract_content_from_file(file):
    if file.filename.endswith('.pdf'):
        return extract_text_from_pdf(file)
    elif file.filename.endswith('.txt'):
        return file.read().decode('utf-8')
    else:
        raise ValueError("Unsupported file type")
    

def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text


def generate_mind_map(content):
    prompt = """You are a professor of cognitive science and an expert in creating detailed mind maps. Your task is to generate a comprehensive mind map in JSON format suitable for the GoJS library from the provided text.
    The mind map should be detailed and include all important points without omitting any details. Capture both the main ideas and the supporting details. Structure the mind map in a hierarchical format, showing relationships and connections between different concepts.
    
    Here is an example of the expected GoJS JSON format for the mind map::
    {{ "class": "go.TreeModel",
        "nodeDataArray": [
            {{"key":0, "text":"Mind Map", "loc":"0 0"}},
            {{"key":1, "parent":0, "text":"Getting more time", "brush":"skyblue", "dir":"right"}},
            {{"key":11, "parent":1, "text":"Wake up early", "brush":"skyblue", "dir":"right"}},
            {{"key":12, "parent":1, "text":"Delegate", "brush":"skyblue", "dir":"right"}},
            {{"key":13, "parent":1, "text":"Simplify", "brush":"skyblue", "dir":"right"}},
            {{"key":2, "parent":0, "text":"More effective use", "brush":"darkseagreen", "dir":"right"}},
            {{"key":21, "parent":2, "text":"Planning", "brush":"darkseagreen", "dir":"right"}},
            {{"key":211, "parent":21, "text":"Priorities", "brush":"darkseagreen", "dir":"right"}},
            {{"key":212, "parent":21, "text":"Ways to focus", "brush":"darkseagreen", "dir":"right"}},
            {{"key":22, "parent":2, "text":"Goals", "brush":"darkseagreen", "dir":"right"}},
            {{"key":3, "parent":0, "text":"Time wasting", "brush":"palevioletred", "dir":"left"}},
            {{"key":31, "parent":3, "text":"Too many meetings", "brush":"palevioletred", "dir":"left"}},
            {{"key":32, "parent":3, "text":"Too much time spent on details", "brush":"palevioletred", "dir":"left"}},
            {{"key":33, "parent":3, "text":"Message fatigue", "brush":"palevioletred", "dir":"left"}},
            {{"key":331, "parent":31, "text":"Check messages less", "brush":"palevioletred", "dir":"left"}},
            {{"key":332, "parent":31, "text":"Message filters", "brush":"palevioletred", "dir":"left"}},
            {{"key":4, "parent":0, "text":"Key issues", "brush":"coral", "dir":"left"}},
            {{"key":41, "parent":4, "text":"Methods", "brush":"coral", "dir":"left"}},
            {{"key":42, "parent":4, "text":"Deadlines", "brush":"coral", "dir":"left"}},
            {{"key":43, "parent":4, "text":"Checkpoints", "brush":"coral", "dir":"left"}}
        ]
    }}

    Text to generate mind map for:
    "{content}"

    Instructions:
    1. **Identify Main Ideas**: Extract the main ideas from the text and place them as primary nodes.
    2. **Capture Supporting Details**: Include all supporting details and sub-points as child nodes under the appropriate main idea.
    3. **Maintain Hierarchical Structure**: Ensure the mind map maintains a clear hierarchical structure.
    4. **Do Not Omit Details**: Ensure no important information is omitted.
    5. **Use Appropriate Coloring**: Use different colors to distinguish between different levels of the hierarchy.

    Output the mind map in the specified JSON format.
    """
    formatted_prompt = textwrap.dedent(prompt).format(content=content)
    logger.debug(f"Formatted prompt: {formatted_prompt}")
    
    model = GenerativeModel(MODEL_ID)
    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 0.3,
        "top_p": 0.95,
    }
    safety_settings = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }
    
    try:
        response = model.generate_content(
            [formatted_prompt],
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        response_text = response.text.replace("```json", "").replace("```", "")
        logger.debug(f"Generated mind map data: {response_text}")
        return response_text
    except Exception as e:
        logger.error(f"Error generating mind map: {e}")
        raise


@app.route('/')
def index():
    logger.debug("Rendering index page")
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    file = request.files['file']
    if file:
        try:
            file_content = extract_content_from_file(file)
            generated_data = generate_mind_map(file_content)
            logger.debug(f"Raw generated data: {generated_data}")
            mind_map_data = json.loads(generated_data)
            logger.debug("Parsed mind map data: {mind_map_data}")
            return jsonify(mind_map_data)
        except Exception as e:
            logger.error(f"Error processing uploaded file: {e}")
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'No file uploaded'}), 400
    

@app.route('/load_map', methods=['POST'])
def load_map():
    file = request.files['file']
    if file:
        try:
            map_data = json.load(file)
            logger.debug(f"Loaded mind map data: {map_data}")
            return jsonify(map_data), 200
        except Exception as e:
            logger.error(f"Error loading mind map: {e}")
            return jsonify({'error': str(e)}), 500
    logger.warning("No file uploaded")
    return jsonify({'error': 'No file uploaded'}), 400


if __name__ == '__main__':
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    logger.debug(f"Initialized Vertex AI with project ID: {PROJECT_ID}, location: {LOCATION}")
    app.run(host='0.0.0.0', port=8080, debug=True)