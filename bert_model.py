import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import string

# Cargar el tokenizer y el modelo BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Verificar si hay una GPU disponible y usarla si es posible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Preprocesar texto
def preprocesar_texto(texto):
    texto = texto.lower()
    texto = re.sub(f"[{re.escape(string.punctuation)}]", "", texto)
    return texto

# Obtener embeddings para un texto
def obtener_embedding(texto):
    texto = preprocesar_texto(texto)
    inputs = tokenizer(texto, return_tensors='pt', truncation=True, padding=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

# Calcular similitud entre embeddings
def calcular_similitud_embedding(respuesta_usuario, respuesta_modelo):
    if not respuesta_usuario.strip():
        return 0.0
    emb_usuario = obtener_embedding(respuesta_usuario)
    emb_modelo = obtener_embedding(respuesta_modelo)
    similitud = cosine_similarity(emb_usuario, emb_modelo)[0][0]
    return similitud

# Evaluar el contexto de la respuesta
def evaluar_contexto(respuesta_usuario, respuesta_modelo):
    longitud_usuario = len(respuesta_usuario.split())
    longitud_modelo = len(respuesta_modelo.split())

    if longitud_usuario < 3:
        return 0.0 

    return 1.0  

def evaluar_respuesta_completa(respuesta_usuario, respuesta_modelo):
    similitud = calcular_similitud_embedding(respuesta_usuario, respuesta_modelo)
    contexto = evaluar_contexto(respuesta_usuario, respuesta_modelo)

    puntaje_total = similitud * contexto
    return puntaje_total

def evaluar_respuesta_exacta(respuesta_usuario, respuesta_modelo):
    prefijos = ["O)", "OQ)","O,"]
    for prefijo in prefijos:
        if respuesta_usuario.startswith(prefijo):
            respuesta_sin_prefijo = respuesta_usuario[len(prefijo):].strip()
            if respuesta_sin_prefijo == respuesta_modelo:
                return 1.0 
            else:
                return 0.0 
    return evaluar_respuesta_completa(respuesta_usuario, respuesta_modelo)
        
    
# Evaluar la respuesta del usuario dependiendo de si empieza con "O)"
def evaluar_respuesta_usuario(respuesta_usuario, respuesta_modelo):
    if respuesta_usuario.startswith("O)") or respuesta_usuario.startswith("OQ)") or  respuesta_usuario.startswith("O") or  respuesta_usuario.startswith("O,"):
        return evaluar_respuesta_exacta(respuesta_usuario, respuesta_modelo)
    else:
        return evaluar_respuesta_completa(respuesta_usuario, respuesta_modelo)