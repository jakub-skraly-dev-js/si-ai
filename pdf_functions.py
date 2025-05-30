import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from pdf2image import convert_from_bytes
import re
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore
import spacy
import nltk
import nltk.data

_original_load = nltk.data.load

def patched_load(resource_name, *args, **kwargs):
    if resource_name == 'tokenizers/punkt_tab/english.pickle':
        resource_name = 'tokenizers/punkt/english.pickle'
    return _original_load(resource_name, *args, **kwargs)

nltk.data.load = patched_load



# Cargar modelo de spacy para español
nlp = spacy.load('es_core_news_sm')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def extract_text_from_pdf(pdf_file, model):
    try:
        pdf_file.seek(0)  
        pages = convert_from_bytes(pdf_file.read(), 300)
        text = ''
        for page_number, page in enumerate(pages):
            print(f"Procesando página {page_number + 1}")
            img = tf.convert_to_tensor(np.array(page))
            img = preprocess_image(img)
            if img is not None:
                img = tf.expand_dims(img, axis=0)
                prediction = model.predict(img)
                print(f"Predicción para página {page_number + 1}: {prediction}")
                if prediction > 0.1:
                    page_text = pytesseract.image_to_string(page, lang='spa')
                    print(f"Texto extraído de la página {page_number + 1}: {page_text}")
                    text += page_text + '\n'
                else:
                    print(f"Predicción inferior a 0.1 para la página {page_number + 1}, se omite el OCR.")
            else:
                print(f"Error al preprocesar la imagen de la página {page_number + 1}.")
        return text
    except Exception as e:
        print(f"Error al procesar el PDF: {str(e)}")
        return ''

def preprocess_image(image):
    try:
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.resnet.preprocess_input(image)
        print(f"Imagen preprocesada: {image.numpy()}")
        return image
    except Exception as e:
        print(f"Error al preprocesar la imagen: {e}")
        return None

def create_model():
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    return model

def extract_names(text):
    lines = text.split('\n')
    first_names = []
    last_names = []
    for line in lines:
        print(f"Procesando línea: {line}")
        if 'Nombres:' in line:
            name = line.replace('Nombres:', '').strip()
            first_names.append(name)
        elif 'Apellidos:' in line:
            surname = line.replace('Apellidos:', '').strip()
            last_names.append(surname)
    print(f"Nombres extraídos: {first_names}, Apellidos extraídos: {last_names}")
    return first_names, last_names

def extract_questions(text):
    question_list = []
    question_counter = 0
    pattern = re.compile(r'\b\d+\.\s.*?[?.;](?=\s|$)', re.IGNORECASE | re.DOTALL)
    matches = pattern.findall(text)

    for match in matches:
        question_list.append(match.strip())
        question_counter += 1

    print(f"Preguntas extraídas: {question_list}, Contador de preguntas: {question_counter}")
    return question_list, question_counter

def extract_points(text):
    points_list = []
    point_counter = 0
    pattern = re.compile(r'\((\d+)\s*puntos?\)', re.IGNORECASE)
    matches = pattern.findall(text)

    for match in matches:
        points_list.append(match.strip())
        point_counter += 1

    print(f"Puntos extraídos: {points_list}, Contador de puntos: {point_counter}")
    return points_list, point_counter

def extract_answers_and_count(text):
    answer_list = []
    answer_count = 0

    pattern = re.compile(r'\d+\.\s.?\?\s\(\d+\spuntos?\)\s*([\s\S]*?)(?=\d+\.\s|\Z)', re.DOTALL)
    matches = pattern.findall(text)

    for match in matches:
        # Buscamos líneas que parezcan opciones o respuestas
        lines = match.strip().split('\n')
        response = ''
        for line in lines:
            if re.match(r'^\s*(O\)|O|@|-|\*)', line.strip()):
                response += line.strip() + ' '
        response = response.strip()
        if response:
            answer_list.append(response)
            answer_count += 1

    print(f"Respuestas extraídas: {answer_list}, Contador de respuestas: {answer_count}")
    return answer_list, answer_count

def preprocess_text(text):
    text = text.replace('\n', ' ')
    return text

def create_json_data(names, questions, points, answers):
    data = {
        "Nombres": names,
        "Preguntas": [{"Pregunta": q, "Puntos": p, "Respuesta": a} for q, p, a in zip(questions, points, answers)]
    }
    json_data = json.dumps(data, ensure_ascii=False, indent=4)
    return json_data
