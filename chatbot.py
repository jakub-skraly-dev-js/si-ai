import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model  # type: ignore

lemmatizer = WordNetLemmatizer()

# Cargar los datos de intents
try:
    intenciones = json.loads(open('fuente_datos.json', 'r', encoding='utf-8').read())
    palabras = pickle.load(open('palabras.pkl', 'rb'))
    clases = pickle.load(open('clase.pkl', 'rb'))
    model = load_model('modelo_entrenado.h5')
except FileNotFoundError:
    print("Error: No se encontraron los archivos necesarios.")
    exit()

def clean_up_sentence(sentence):
    sentence_palabras = nltk.word_tokenize(sentence)
    sentence_palabras = [lemmatizer.lemmatize(word.lower()) for word in sentence_palabras]
    return sentence_palabras

def bag_of_palabras(sentence):
    sentence_palabras = clean_up_sentence(sentence)
    bag = [0]*len(palabras)
    for w in sentence_palabras:
        for i, word in enumerate(palabras):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_palabras(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': clases[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intenciones'] 
    for intent in list_of_intents:
        if intent['tag'] == tag:
            result = random.choice(intent['responses'])
            break
    return result
