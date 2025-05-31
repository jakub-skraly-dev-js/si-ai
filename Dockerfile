FROM python:3.10-slim

# Instala dependencias del sistema y Tesseract
RUN apt-get update && \
    apt-get install -y tesseract-ocr tesseract-ocr-spa poppler-utils gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copia los archivos del proyecto
WORKDIR /app
COPY . /app

# Instala las dependencias de Python
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader punkt punkt_tab wordnet averaged_perceptron_tagger
RUN python bert_model.py
RUN python -m spacy download es_core_news_sm

# # Descarga el modelo de spaCy para español
# RUN python -m spacy download es_core_news_sm

# # Descarga recursos de NLTK usando el script
# RUN python download_nltk_data.py

# Comando por defecto (ajusta según tu entrypoint real)
# CMD ["python", "Entrenamiento.py"]
# CMD ["python", "bert_model.py"]
# CMD ["python", "pdf_functions.py"]
# CMD ["python", "app.py"]

# RUN python Entrenamiento.py
# RUN python bert_model.py
# RUN python pdf_functions.py

CMD ["python", "app.py"]