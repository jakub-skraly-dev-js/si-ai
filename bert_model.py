# bert_model.py

import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import string

# Cargar tokenizer y modelo
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Preprocesamiento
def preprocesar_texto(texto):
    texto = texto.lower()
    texto = re.sub(f"[{re.escape(string.punctuation)}]", "", texto)
    return texto

# Embeddings
def obtener_embedding(texto):
    texto = preprocesar_texto(texto)
    inputs = tokenizer(texto, return_tensors='pt', truncation=True, padding=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

# Similitud por BERT
def calcular_similitud_embedding(respuesta_usuario, respuesta_modelo):
    if not respuesta_usuario.strip():
        return 0.0
    emb_usuario = obtener_embedding(respuesta_usuario)
    emb_modelo = obtener_embedding(respuesta_modelo)
    return cosine_similarity(emb_usuario, emb_modelo)[0][0]

# Evaluar contexto (relajado)
def evaluar_contexto(respuesta_usuario, respuesta_modelo):
    longitud_usuario = len(respuesta_usuario.split())
    if longitud_usuario == 0:
        return 0.0
    elif longitud_usuario < 3:
        return 0.8  # Penalización leve
    return 1.0

# Evaluar respuestas largas o abiertas
def evaluar_respuesta_completa(respuesta_usuario, respuesta_modelo):
    similitud = calcular_similitud_embedding(respuesta_usuario, respuesta_modelo)
    contexto = evaluar_contexto(respuesta_usuario, respuesta_modelo)
    return similitud * contexto

# Evaluar opción múltiple (flexible)
def evaluar_respuesta_exacta(respuesta_usuario, respuesta_modelo):
    prefijos = ["O)", "OQ)", "O,", "O"]
    respuesta_usuario = respuesta_usuario.strip()
    for prefijo in prefijos:
        if respuesta_usuario.startswith(prefijo):
            respuesta_sin_prefijo = preprocesar_texto(respuesta_usuario[len(prefijo):].strip())
            respuesta_modelo = preprocesar_texto(respuesta_modelo.strip())
            return 1.0 if respuesta_sin_prefijo == respuesta_modelo else 0.0
    return evaluar_respuesta_completa(respuesta_usuario, respuesta_modelo)

# Wrapper principal
def evaluar_respuesta_usuario(respuesta_usuario, respuesta_modelo):
    prefijos = ["O)", "OQ)", "O,", "O"]
    if any(respuesta_usuario.strip().startswith(p) for p in prefijos):
        return evaluar_respuesta_exacta(respuesta_usuario, respuesta_modelo)
    else:
        return evaluar_respuesta_completa(respuesta_usuario, respuesta_modelo)
