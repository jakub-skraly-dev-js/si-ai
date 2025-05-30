import os
import base64
from io import BytesIO
from flask_cors import CORS
from chatbot import predict_class, get_response, intenciones  # Ajusta la importación según corresponda
from bert_model import evaluar_respuesta_usuario  # Importa la función de evaluación
from flask import Flask, request, jsonify
import pytesseract

from pdf_functions import (
    extract_text_from_pdf,
    extract_names,
    extract_questions,
    extract_points,
    extract_answers_and_count,
    preprocess_text,
    create_model,
    create_json_data
)
import tensorflow as tf


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "https://si-front-black.vercel.app"}})  

model = create_model()

@app.route('/')
def index():
    return 'Hello, World!'



@app.route('/evaluar', methods=['GET', 'POST'])
def evaluar():
    if request.method == 'GET':
        return jsonify({"message": "Usa POST con uno o más PDFs en base64 para evaluar."})

    try:
        content = request.get_json()
        pdf_base64_list = content.get('pdf_base64', [])

        if not pdf_base64_list:
            return jsonify({"error": "No se ha encontrado ningún dato base64 de PDF."}), 400

        resultados_pdf = []

        for idx, pdf_base64 in enumerate(pdf_base64_list):
            try:
                pdf_data = base64.b64decode(pdf_base64)
                pdf_file = BytesIO(pdf_data)
                pdf_text = extract_text_from_pdf(pdf_file, model)
                print(f"Texto extraído del PDF {idx + 1}: {pdf_text}")

                json_content = process_and_return(pdf_text, idx + 1)

                preguntas = json_content['json_content']['Preguntas']
                respuestas_usuario = json_content['json_content']['Respuestas']
                pesos = json_content['json_content']['Puntos']

                respuestas_modelo = [obtener_respuesta_chatbot(p) for p in preguntas]
                puntajes = evaluar_examenes(respuestas_usuario, respuestas_modelo, pesos)

                json_content['json_content']['PuntajesPorPregunta'] = puntajes['puntajes_por_pregunta']
                json_content['json_content']['PuntajeTotal'] = puntajes['puntaje_total']

                resultados_pdf.append(json_content)

            except Exception as e:
                print(f"Error al procesar archivo base64 {idx + 1}: {e}")
                resultados_pdf.append({"error": f"Error al procesar archivo base64 {idx + 1}: {e}"})

        return jsonify({"resultados_pdf": resultados_pdf})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def process_and_return(text, pdf_index):
    extracted_names = extract_names(text)
    extracted_questions, _ = extract_questions(text)
    extracted_points, _ = extract_points(text)
    extracted_answers, _ = extract_answers_and_count(text)

    json_content = {
        "pdf_index": pdf_index,
        "json_content": {
            "Nombres": [list(name) for name in extracted_names],
            "Preguntas": extracted_questions,
            "Respuestas": extracted_answers,
            "Puntos": extracted_points
        }
    }

    return json_content

def obtener_respuesta_chatbot(pregunta):
    intents_list = predict_class(pregunta)
    response = get_response(intents_list, intenciones)
    return response

def evaluar_examenes(respuestas_usuario, respuestas_modelo, pesos):
    puntajes_por_pregunta = []
    puntaje_total = 0
    
    for respuesta_usuario, respuesta_modelo, peso in zip(respuestas_usuario, respuestas_modelo, pesos):
        peso_numero = int(peso.split()[0]) 
        
        puntaje = evaluar_respuesta_usuario(respuesta_usuario, respuesta_modelo)
        puntaje_ponderado = puntaje * peso_numero
        puntajes_por_pregunta.append(puntaje_ponderado)
        puntaje_total += puntaje_ponderado
    
    return {
        "puntajes_por_pregunta": puntajes_por_pregunta,
        "puntaje_total": puntaje_total
    }

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(debug=True, host='0.0.0.0', port=port)
