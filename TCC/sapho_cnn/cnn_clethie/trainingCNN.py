import os
import sys
import numpy as np
import tensorflow as tf

# ==================== CARREGAR DADOS ====================
# Caminhos relativos a este arquivo: funciona em qualquer PC.
# O dataset fica em ManipulacaoDados/ (fora do git, ver .gitignore).
# Para usar outro arquivo:  python trainingCNN.py caminho/para/dataset_completo.npz
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    BASE_DIR, "ManipulacaoDados", "DadosConcatenados", "janelamento_7", "dataset_completo.npz")
if not os.path.exists(dataset_path):
    print(f"Nao achei o dataset: {dataset_path}\n>>> Coloque dataset_completo.npz nesse caminho ou passe o caminho como argumento.")
    raise SystemExit(1)

data = np.load(dataset_path)
MatrizAmostras = data['MatrizAmostras']
AmplitudeAssociada = data['AmplitudeReal']
Ocupacao = data['Ocupacao']

N_JANELAMENTO = 7
VALIDATION_SPLIT = 0.1
BATCH_SIZE = 4096  
EPOCHS = 15

# ==================== MODELO ====================
def criar_cnn():
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((N_JANELAMENTO, 1), input_shape=(N_JANELAMENTO,)),
        tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu', padding='same', input_shape=(N_JANELAMENTO, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# ==================== TREINAR ====================
print("Treinando modelo...")
modelo = criar_cnn()
modelo.summary()

history = modelo.fit(
    MatrizAmostras, AmplitudeAssociada,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    verbose=1
)

# Salvar o modelo
modelo.save(os.path.join(BASE_DIR, 'cnn_trained.keras'))
modelo.save(os.path.join(BASE_DIR, 'cnn_trained.h5'))
print(f"Modelo salvo em {os.path.join(BASE_DIR, 'cnn_trained.h5')}")