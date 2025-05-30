import nltk

# Lista de recursos comúnmente utilizados
resources = [
    'punkt',
    'punkt_tab', 
    'averaged_perceptron_tagger',
    'wordnet',
    'stopwords',
    'vader_lexicon',
    'omw-1.4',
    'maxent_ne_chunker',
    'words'
]

for resource in resources:
    try:
        nltk.download(resource, quiet=True)
        print(f"✓ Descargado: {resource}")
    except Exception as e:
        print(f"✗ Error descargando {resource}: {e}")

print("Descarga de recursos NLTK completada")