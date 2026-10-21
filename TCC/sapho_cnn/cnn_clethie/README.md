# CNN FPGA Pipeline

> Pipeline completo para treinar uma CNN em Python, exportar arquitetura e pesos para C, e executar inferência diretamente em C — com foco em sistemas embarcados e FPGA.

---

## 📌 Visão Geral

A proposta do projeto é permitir que uma CNN treinada em alto nível (Python) seja convertida para uma implementação em baixo nível (C), possibilitando:

- Execução eficiente sem dependência de frameworks como TensorFlow/PyTorch
- Portabilidade para sistemas embarcados e microcontroladores
- Integração futura com FPGA via síntese de alto nível (HLS)

**Fluxo principal:**

```
Dados (N×7) ──► Treinamento Python ──► Exportação de Pesos ──► Inferência em C
```

Os dados de entrada são organizados como:
- Matrizes de dimensão **N × 7** (séries temporais ou features por amostra)
- Vetor de valores reais como **targets** (regressão)

---

## 🗂️ Estrutura do Projeto

```
.
├── cnn_weights_c/          # Código C gerado e pesos exportados
│   ├── cnn_inference.c     # Implementação da inferência em C
│   ├── conv1_weights.h     # Pesos da camada conv1
│   ├── conv2_weights.h     # Pesos da camada conv2
│   ├── bn1_weights.h       # Parâmetros do BatchNorm 1
│   ├── bn2_weights.h       # Parâmetros do BatchNorm 2
│   ├── dense1_weights.h    # Pesos da camada densa 1
│   ├── dense2_weights.h    # Pesos da camada densa 2
│   └── dense3_weights.h    # Pesos da camada de saída
│
├── ManipulacaoDados/        # Scripts de pré-processamento
│
├── trainingCNN.py           # Treinamento da CNN em Python
├── export_weights.py        # Exporta pesos e arquitetura para C
├── .gitignore
└── README.md
```

---

## 📊 Arquitetura da CNN

A rede neural convolucional implementada possui a seguinte estrutura:

| Camada       | Tipo                  | Arquivo de Pesos     |
|--------------|-----------------------|----------------------|
| `conv1`      | Convolucional 1D      | `conv1_weights.h`    |
| `bn1`        | Batch Normalization   | `bn1_weights.h`      |
| `conv2`      | Convolucional 1D      | `conv2_weights.h`    |
| `bn2`        | Batch Normalization   | `bn2_weights.h`      |
| `dense1`     | Camada Densa          | `dense1_weights.h`   |
| `dense2`     | Camada Densa          | `dense2_weights.h`   |
| `dense3`     | Saída (regressão)     | `dense3_weights.h`   |

Os pesos são exportados como arquivos `.h` com arrays estáticos e utilizados diretamente no código C via `#include`, sem necessidade de leitura em tempo de execução.

---

## ⚙️ Pipeline de Execução

### 1. Preparar os dados

Organize os dados de treinamento no formato esperado:

- Matrizes de dimensão `N × 7` (N amostras, 7 features cada)
- Vetor de targets com N valores reais
- Pré-processamento disponível em `ManipulacaoDados/`

### 2. Treinar a CNN e exportar os pesos para C

```bash
python export_weights.py
```

Isso irá gerar dentro de `cnn_weights_c/`:
- Arquivos `.h` com os pesos de cada camada em arrays C estáticos
- O arquivo `cnn_inference.c` com a lógica de inferência completa

> ⚠️ Adapte `export_weights.py` caso utilize um formato de entrada diferente (ex: imagens) ou uma estratégia de normalização distinta.

### 3. Compilar o código C

Dentro da pasta `cnn_weights_c/`:

```bash
gcc -O2 -lm -o cnn_inference cnn_inference.c
```

### 4. Executar a inferência

```bash
./cnn_inference
```

A saída será o resultado da inferência para os dados de entrada fornecidos.

---

## 🔢 Formato dos Dados

| Propriedade      | Valor                        |
|------------------|------------------------------|
| Entrada          | Matriz `N × 7` (float)       |
| Saída            | Vetor de `N` valores reais   |
| Tarefa           | Regressão                    |
| Normalização     | Definida em `export_weights.py` |

O código C assume que os dados de entrada já estão normalizados da mesma forma que durante o treinamento. Certifique-se de aplicar a mesma normalização antes de passar dados para a inferência em C.

---

## ⚠️ Observações Importantes

- O código C assume um **formato fixo de entrada** — adaptações podem ser necessárias para diferentes dimensões ou tipos de dados
- A **normalização dos dados** deve ser replicada exatamente no lado C para garantir resultados corretos
- O arquivo `export_weights.py` deve ser revisado ao mudar a arquitetura da rede
- Para uso em FPGA, considere ajustes de precisão numérica (ponto fixo vs. ponto flutuante)

---

## 🚀 Possíveis Extensões

- **Integração com FPGA** via High-Level Synthesis (HLS) — ex: Xilinx Vitis HLS
- **Quantização de pesos** para `int8` ou ponto fixo, reduzindo área e consumo de energia
- **Otimizações de memória** — loop unrolling, pipelining, tiling
- **Suporte a outras arquiteturas** — LSTM, Transformer, MLP puro
- **Benchmark** de desempenho C vs. Python para validação da exportação

---

## 🧪 Validação

Para garantir que a exportação está correta, recomenda-se comparar as saídas:

1. Rodar inferência no modelo Python (TensorFlow/Keras) com amostras de teste
2. Rodar `cnn_inference` em C com as mesmas amostras
3. Comparar os valores de saída — diferenças devem ser negligenciáveis (ordem de `1e-5`)

---

## 🛠️ Dependências

### Python
- `TensorFlow` / `Keras`
- `NumPy`
- `Pandas` (para pré-processamento, se aplicável)

### C
- Compilador GCC (ou equivalente)
- Biblioteca matemática padrão (`-lm`)
- Nenhuma dependência externa — apenas headers gerados pelo `export_weights.py`

---

## 📌 Objetivo

Este projeto serve como base para:

- Inferência embarcada de redes neurais sem frameworks de alto nível
- Conversão de modelos Python para implementações C portáveis
- Pesquisa em aceleração de redes neurais em hardware dedicado (FPGA/ASIC)

---

## 👨‍💻 Autor

**Diogo Cardinot**  
Mestrado em Engenharia de Computação