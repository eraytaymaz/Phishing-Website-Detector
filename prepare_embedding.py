import os
import sys
import torch
import pickle
import numpy as np
from trafilatura import extract
from googletrans import Translator
from sentence_transformers import SentenceTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def parse_html(folder_path):
    parsed_list = list()
    label = 0 if folder_path == "Legitimate" else 1
    for html in os.listdir(folder_path):
        html_path = os.path.join(folder_path, html)
        with open(html_path, 'r', encoding=find_encoding(html_path)) as file:
            html_content = file.read()
        parsed = extract(html_content)
        if parsed != None: parsed_list.append((parsed, label))
    return parsed_list

def choose_model(model):
    if model == "xlm-roberta":
        return SentenceTransformer('aditeyabaral/sentencetransformer-xlm-roberta-base')
    elif model == "sbert":
        return SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')

def decide_translate(model):
    if model == "xlm-roberta": return False
    return True

def create_embeddings(train_data, model, b_translate):
    embeddings_list = list()
    translator = Translator()
    for data in train_data:
        content = data[0]
        if b_translate:
            if len(content) < 5000: content = translator.translate(content).text
            else: continue
        embedding = model.encode(content, show_progress_bar=True)
        embeddings_list.append((embedding, data[1]))
    return embeddings_list

def save_embeddings(model, train_embeddings):
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    path = "embeddings/embeddings-"+model+".pkl"
    with open(path, 'wb') as file:
        pickle.dump(train_embeddings, file)


train_legitimate = parse_html("Legitimate")

train_phishing = parse_html("Phishing")

transformer_model = sys.argv[1][1:]

b_translate = decide_translate(transformer_model)

model = choose_model(transformer_model)
model = model.to(device)

legitimate_embeddings = create_embeddings(train_legitimate, model, b_translate)

phishing_embeddings = create_embeddings(train_phishing, model, b_translate)

train_embeddings = legitimate_embeddings + phishing_embeddings

save_embeddings(transformer_model, train_embeddings)