import joblib
from flask import Flask, render_template, request
from trafilatura import extract
from sentence_transformers import SentenceTransformer
 
app = Flask(__name__)

model = joblib.load('model/xgboost_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'htmlFile' not in request.files:
        return "No file part"
    
    file = request.files['htmlFile']

    if file.filename == '':
        return "No selected file"

    file_path = 'test/' + file.filename
    file.save(file_path)

    parsed = parse_html(file_path)
    embedding = transform2vector(parsed)
    testing = list()
    testing.append(embedding)
    prediction_result = "Phishing" if model.predict(testing) else "Legitimate"
    
    return f"{file_path} is {prediction_result}"


def find_encoding(html_path):
    try:
        with open(html_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        encoding_used = 'utf-8'
    except UnicodeDecodeError:
        with open(html_path, 'r', encoding='windows-1256') as file:
            html_content = file.read()
        encoding_used = 'windows-1256'
    return encoding_used

def parse_html(html_path):
    with open(html_path, 'r', encoding=find_encoding(html_path)) as file:
        html_content = file.read()
    parsed = extract(html_content)
    return parsed

def transform2vector(parsed):
    model = SentenceTransformer('aditeyabaral/sentencetransformer-xlm-roberta-base')
    embedding = model.encode(parsed)
    return embedding


if __name__ == '__main__':
    app.run(debug=True, port=5050)