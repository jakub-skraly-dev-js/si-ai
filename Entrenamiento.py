import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np 
from keras.models import Sequential # type: ignore
from keras.layers import Dense, Dropout # type: ignore
from keras.optimizers import SGD # type: ignore
import random

nltk.download('punkt')
nltk.download('wordnet')

data_file = open('fuente_datos.json', 'r', encoding='utf-8').read()
intenciones = json.loads(data_file)

lemmatizer = WordNetLemmatizer()

palabras = []
clases = []
documentos = []
ignorar_pala = ['?', '1']

for intent in intenciones['intenciones']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        palabras.extend(w)
        documentos.append((w, intent['tag']))
        if intent['tag'] not in clases:
            clases.append(intent['tag'])

palabras = [lemmatizer.lemmatize(w.lower()) for w in palabras if w not in ignorar_pala]
palabras = sorted(list(set(palabras)))
clases = sorted(list(set(clases)))

# Guardar palabras y clases en archivos pickle
pickle.dump(palabras, open('palabras.pkl', 'wb'))
pickle.dump(clases, open('clase.pkl', 'wb'))

# Crear dataset de entrenamiento
training = []
output_empty = [0] * len(clases)

for doc in documentos:
    bag = []
    pattern_palabras = doc[0]
    pattern_palabras = [lemmatizer.lemmatize(word.lower()) for word in pattern_palabras]

    for word in palabras:
        bag.append(1) if word in pattern_palabras else bag.append(0)

    output_row = list(output_empty)
    output_row[clases.index(doc[1])] = 1

    training.append([bag, output_row])

# Mezclar los datos de entrenamiento
random.shuffle(training)

# Convertir datos a arrays numpy
train_x = np.array([row[0] for row in training])
train_y = np.array([row[1] for row in training])

# Crear modelo de red neuronal
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Configurar el optimizador SGD
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Entrenar el modelo
hist = model.fit(train_x, train_y, epochs=300, batch_size=5, verbose=1)

# Guardar el modelo entrenado
model.save('modelo_entrenado.h5')

print("Modelo creado y guardado como modelo_entrenado.h5")
