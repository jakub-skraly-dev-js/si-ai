import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

from pdf2image import convert_from_bytes
import re
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model  # type: ignore
import spacy
import nltk

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
                # Procesar todas las páginas sin importar el valor de predicción
                page_text = pytesseract.image_to_string(page, lang='spa')
                print(f"Texto extraído de la página {page_number + 1}: {page_text}")
                text += page_text + '\n'
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
        if 'Nombres:' in line:
            name = line.replace('Nombres:', '').strip()
            first_names.append(name)
        elif 'Apellidos:' in line:
            surname = line.replace('Apellidos:', '').strip()
            last_names.append(surname)
    return first_names, last_names


def extract_questions(text):
    pattern = re.compile(r'\b\d+\.\s.*?[?.;](?=\s|$)', re.IGNORECASE | re.DOTALL)
    matches = pattern.findall(text)
    return [match.strip() for match in matches], len(matches)


def extract_points(text):
    pattern = re.compile(r'\((\d+)\s*puntos?\)', re.IGNORECASE)
    matches = pattern.findall(text)
    return [match.strip() for match in matches], len(matches)


def extract_answers_and_count(text):
    answer_list = []
    pattern = re.compile(r'\d+\.\s.*?\(\d+\spuntos?\)\s*([\s\S]*?)(?=\n\d+\.\s|\Z)', re.DOTALL)
    matches = pattern.findall(text)

    for match in matches:
        lines = match.strip().split('\n')
        response_lines = []
        for line in lines:
            if re.match(r'^\s*(O\)|O\.|@|-|\*|\([a-dA-D]\)|[a-dA-D]\))\s+', line) or line.strip():
                response_lines.append(line.strip())
        response = ' '.join(response_lines).strip()
        answer_list.append(response)

    return answer_list, len(answer_list)


def preprocess_text(text):
    return text.replace('\n', ' ')


def create_json_data(names, questions, points, answers):
    from itertools import zip_longest
    data = {
        "Nombres": names,
        "Preguntas": [
            {
                "Pregunta": q or "",
                "Puntos": p or "0",
                "Respuesta": a or ""
            }
            for q, p, a in zip_longest(questions, points, answers, fillvalue="")
        ]
    }
    return json.dumps(data, ensure_ascii=False, indent=4)
