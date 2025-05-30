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

# Descarga el modelo de spaCy para español
RUN python -m spacy download es_core_news_sm

RUN python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"

# Comando por defecto (ajusta según tu entrypoint real)
CMD ["python", "app.py"]