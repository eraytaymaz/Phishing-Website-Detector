import os
import sys
import torch
import pickle
import numpy as np
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_type = sys.argv[1][1:]
embedding_path = sys.argv[2][1:]

def choose_model(model_type):
    if model_type == "xgb": return XGBClassifier(n_estimators=1000, max_depth=5, learning_rate=0.1, objective='binary:logistic',tree_method = "hist", device = "cuda")
    elif model_type == "cat": return CatBoostClassifier(iterations=1000, task_type="GPU", devices='0:1')
    elif model_type == "ann": return MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=1000)

def load_embeddings(embedding_path):
    with open(embedding_path, 'rb') as file:
        embeddings_list = pickle.load(file)
    return embeddings_list

def split_data_label(train_data, choose):
    data_or_label = list()
    for i in range(len(train_data)):
        data_or_label.append(train_data[i][choose])
    return data_or_label

def get_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")

def save_model(model, model_type):
    if not os.path.exists("model"):
        os.makedirs("model")
    
    if model_type == "xgb": model_type += "oost"
    elif model_type == "cat": model_type += "boost"
            
    path = "model/"+model_type+"_model.pkl"
    with open(path, 'wb') as file:
        pickle.dump(model, file)

model = choose_model(model_type)
embeddings_list = load_embeddings(embedding_path)
sentences = split_data_label(embeddings_list, 0)
labels = split_data_label(embeddings_list, 1)

X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2)

if model_type == "xgb":
    model.fit(X_train, y_train)
elif model_type == "cat":
    model.fit(X_train, y_train,verbose=False)
elif model_type == "ann":
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model.fit(X_train, y_train)

get_metrics(model, X_test, y_test)

save_model(model, model_type)