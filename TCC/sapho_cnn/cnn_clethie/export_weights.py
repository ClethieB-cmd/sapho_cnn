import os
import sys
import numpy as np
import tensorflow as tf

# ==================== CONFIGURAÇÕES ====================
# Caminhos relativos a este arquivo: funciona em qualquer PC.
# O dataset fica em ManipulacaoDados/ (fora do git, ver .gitignore).
# Para usar outro arquivo:  python export_weights.py caminho/para/dataset_completo.npz
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    BASE_DIR, "ManipulacaoDados", "DadosConcatenados", "janelamento_7", "dataset_completo.npz")
if not os.path.exists(dataset_path):
    print(f"Nao achei o dataset: {dataset_path}\n>>> Coloque dataset_completo.npz nesse caminho ou passe o caminho como argumento.")
    raise SystemExit(1)
OUTPUT_DIR = os.path.join(BASE_DIR, "cnn_weights_c")

N_JANELAMENTO = 7
VALIDATION_SPLIT = 0.1
BATCH_SIZE = 4096
EPOCHS = 15

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== CARREGAR DADOS ====================
print("Carregando dados...")
data = np.load(dataset_path)
MatrizAmostras = data['MatrizAmostras'].astype(np.float32)
AmplitudeAssociada = data['AmplitudeReal'].astype(np.float32)

# ==================== MODELO ====================
def criar_cnn():
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((N_JANELAMENTO, 1), input_shape=(N_JANELAMENTO,)),
        tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu', padding='same'),
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

print(f"\nMAE final treino:     {history.history['mae'][-1]:.4f}")
print(f"MAE final validação:  {history.history['val_mae'][-1]:.4f}")

# ==================== EXTRAIR PESOS ====================
# Mapear camadas (ignorar Reshape e Dropout, que não têm pesos)
layer_map = {}
for layer in modelo.layers:
    w = layer.get_weights()
    if len(w) > 0:
        layer_map[layer.name] = {'type': type(layer).__name__, 'weights': w}
        print(f"  Camada '{layer.name}' ({type(layer).__name__}): {[x.shape for x in w]}")

# ==================== HELPER: array C ====================
def array_to_c(arr, name, dtype="float"):
    """Converte numpy array para declaração de array C."""
    flat = arr.flatten()
    shape = arr.shape
    total = flat.size

    # Montar dimensões para o nome do array
    dims = "".join(f"[{d}]" for d in shape)
    lines = [f"static const {dtype} {name}{dims} = {{"]

    # Escrever valores em linhas de 8
    vals = []
    for i, v in enumerate(flat):
        vals.append(f"{v:.8f}f")
        if (i + 1) % 8 == 0 or i == total - 1:
            lines.append("    " + ", ".join(vals) + ",")
            vals = []

    lines.append("};")
    return "\n".join(lines)

# ==================== GERAR HEADERS ====================

# ---------- Conv1 ----------
conv1_layer = modelo.get_layer(index=1)   # Conv1D 32 filtros
w_conv1, b_conv1 = conv1_layer.get_weights()
# w_conv1 shape: (kernel_size=5, in_channels=1, out_channels=32)
# Transpor para (out_channels, kernel_size, in_channels) — mais intuitivo em C
w_conv1_t = np.transpose(w_conv1, (2, 0, 1))  # (32, 5, 1)

with open(os.path.join(OUTPUT_DIR, "conv1_weights.h"), "w") as f:
    f.write("#pragma once\n\n")
    f.write("/* Conv1D camada 1: 32 filtros, kernel=5, in_ch=1 */\n")
    f.write("/* Layout: [filtro][posicao_kernel][canal_entrada] */\n\n")
    f.write("#define CONV1_FILTERS   32\n")
    f.write("#define CONV1_KERNEL    5\n")
    f.write("#define CONV1_IN_CH     1\n\n")
    f.write(array_to_c(w_conv1_t, "conv1_kernel"))
    f.write("\n\n")
    f.write(array_to_c(b_conv1, "conv1_bias"))
    f.write("\n")

print("  -> conv1_weights.h gerado")

# ---------- BN1 ----------
bn1_layer = modelo.get_layer(index=2)
gamma1, beta1, mean1, var1 = bn1_layer.get_weights()

with open(os.path.join(OUTPUT_DIR, "bn1_weights.h"), "w") as f:
    f.write("#pragma once\n\n")
    f.write("/* BatchNormalization camada 1 (32 canais) */\n\n")
    f.write("#define BN1_SIZE 32\n\n")
    f.write(array_to_c(gamma1, "bn1_gamma"))
    f.write("\n\n")
    f.write(array_to_c(beta1,  "bn1_beta"))
    f.write("\n\n")
    f.write(array_to_c(mean1,  "bn1_mean"))
    f.write("\n\n")
    f.write(array_to_c(var1,   "bn1_var"))
    f.write("\n")

print("  -> bn1_weights.h gerado")

# ---------- Conv2 ----------
conv2_layer = modelo.get_layer(index=3)
w_conv2, b_conv2 = conv2_layer.get_weights()
# w_conv2 shape: (kernel_size=3, in_channels=32, out_channels=16)
w_conv2_t = np.transpose(w_conv2, (2, 0, 1))  # (16, 3, 32)

with open(os.path.join(OUTPUT_DIR, "conv2_weights.h"), "w") as f:
    f.write("#pragma once\n\n")
    f.write("/* Conv1D camada 2: 16 filtros, kernel=3, in_ch=32 */\n")
    f.write("/* Layout: [filtro][posicao_kernel][canal_entrada] */\n\n")
    f.write("#define CONV2_FILTERS   16\n")
    f.write("#define CONV2_KERNEL    3\n")
    f.write("#define CONV2_IN_CH     32\n\n")
    f.write(array_to_c(w_conv2_t, "conv2_kernel"))
    f.write("\n\n")
    f.write(array_to_c(b_conv2, "conv2_bias"))
    f.write("\n")

print("  -> conv2_weights.h gerado")

# ---------- BN2 ----------
bn2_layer = modelo.get_layer(index=4)
gamma2, beta2, mean2, var2 = bn2_layer.get_weights()

with open(os.path.join(OUTPUT_DIR, "bn2_weights.h"), "w") as f:
    f.write("#pragma once\n\n")
    f.write("/* BatchNormalization camada 2 (16 canais) */\n\n")
    f.write("#define BN2_SIZE 16\n\n")
    f.write(array_to_c(gamma2, "bn2_gamma"))
    f.write("\n\n")
    f.write(array_to_c(beta2,  "bn2_beta"))
    f.write("\n\n")
    f.write(array_to_c(mean2,  "bn2_mean"))
    f.write("\n\n")
    f.write(array_to_c(var2,   "bn2_var"))
    f.write("\n")

print("  -> bn2_weights.h gerado")

# ---------- Dense1 (64) ----------
dense1_layer = modelo.get_layer(index=6)   # índice 5=Flatten (sem pesos), 6=Dense64
w_d1, b_d1 = dense1_layer.get_weights()
# w shape: (in, out) = (7*16=112, 64)

with open(os.path.join(OUTPUT_DIR, "dense1_weights.h"), "w") as f:
    f.write("#pragma once\n\n")
    f.write("/* Dense camada 1: 112 -> 64, ReLU */\n\n")
    f.write(f"#define DENSE1_IN   {w_d1.shape[0]}\n")
    f.write(f"#define DENSE1_OUT  {w_d1.shape[1]}\n\n")
    f.write(array_to_c(w_d1, "dense1_W"))
    f.write("\n\n")
    f.write(array_to_c(b_d1, "dense1_b"))
    f.write("\n")

print("  -> dense1_weights.h gerado")

# ---------- Dense2 (32) ----------
dense2_layer = modelo.get_layer(index=8)   # índice 7=Dropout, 8=Dense32
w_d2, b_d2 = dense2_layer.get_weights()

with open(os.path.join(OUTPUT_DIR, "dense2_weights.h"), "w") as f:
    f.write("#pragma once\n\n")
    f.write("/* Dense camada 2: 64 -> 32, ReLU */\n\n")
    f.write(f"#define DENSE2_IN   {w_d2.shape[0]}\n")
    f.write(f"#define DENSE2_OUT  {w_d2.shape[1]}\n\n")
    f.write(array_to_c(w_d2, "dense2_W"))
    f.write("\n\n")
    f.write(array_to_c(b_d2, "dense2_b"))
    f.write("\n")

print("  -> dense2_weights.h gerado")

# ---------- Dense3 / Output (1) ----------
dense3_layer = modelo.get_layer(index=10)  # índice 9=Dropout, 10=Dense1
w_d3, b_d3 = dense3_layer.get_weights()

with open(os.path.join(OUTPUT_DIR, "dense3_weights.h"), "w") as f:
    f.write("#pragma once\n\n")
    f.write("/* Dense camada 3 (saida): 32 -> 1, linear */\n\n")
    f.write(f"#define DENSE3_IN   {w_d3.shape[0]}\n")
    f.write(f"#define DENSE3_OUT  {w_d3.shape[1]}\n\n")
    f.write(array_to_c(w_d3, "dense3_W"))
    f.write("\n\n")
    f.write(array_to_c(b_d3, "dense3_b"))
    f.write("\n")

print("  -> dense3_weights.h gerado")

# ==================== VERIFICAÇÃO: comparar saídas Python vs exportado ====================
print("\n=== VERIFICAÇÃO DE SANIDADE ===")
n_check = 5
amostra = MatrizAmostras[:n_check]
pred_keras = modelo.predict(amostra, verbose=0).flatten()
print("Predições Keras (referência):")
for i, v in enumerate(pred_keras):
    print(f"  Amostra {i}: {v:.6f}")

print(f"\nPesos exportados em: {OUTPUT_DIR}/")
print("Arquivos gerados:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    path_f = os.path.join(OUTPUT_DIR, f)
    size_kb = os.path.getsize(path_f) / 1024
    print(f"  {f:30s}  ({size_kb:.1f} KB)")

print("\nDone! Agora compile cnn_inference.c com os headers gerados.")