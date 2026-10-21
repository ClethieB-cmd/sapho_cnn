# sapho_cnn — CNN treinada rodando no soft-processor SAPHO

Valida que o processador SAPHO **`cnn_sapho`** reproduz a saída da **CNN treinada**
(repositório `RedeNeuralConvolucionalFPGA`) para as mesmas entradas.

A rede é a CNN completa, traduzida para C± e executada como programa no processador:
`Conv1D(32,k5)→ReLU→BN1 → Conv1D(16,k3)→ReLU→BN2 → Flatten(112) → Dense(64)→ReLU → Dense(32)→ReLU → Dense(1)`.

## Conteúdo
| Arquivo | O que é |
|---|---|
| `cnn_sapho/cnn_sapho.cmm` | a CNN inteira em C± |
| `cnn_sapho/*_W.txt`, `*_b.txt`, `bn*_a/b.txt` (14) | pesos **treinados** + BatchNorm (estáticos — não precisa regenerar) |
| `cnn_sapho/cnn_sapho/` | processador gerado pelo SAPHO para este programa (`.v`, `.mif`, `.asm`); recriado a cada compilação |
| `cnn_sapho/test_input_sapho.txt` | 20 pulsos reais × 7 amostras, escalados ×2²⁰ |
| `cnn_sapho/test_expected.txt` | gabarito = saída da CNN Keras (`AmplitudeEstimada`) |
| `cnn_sapho_top.v` / `cnn_sapho_test.v` | top-level e testbench |
| `compare.py` | compara saída do SAPHO × gabarito |
| `sapho_cnn.spf` | projeto Aurora (só o processador `cnn_sapho`) |

## Como rodar (Aurora/SAPHO) em qualquer PC
1. Abra `sapho_cnn.spf` no Aurora (ele ajusta os caminhos do projeto sozinho).
2. **Ajuste 1 linha:** em `cnn_sapho_test.v`, edite o ``\`define PROJ`` para o caminho da pasta
   `.../sapho_cnn/cnn_sapho` **neste PC** (use barras `/`).
3. Compile o processador **`cnn_sapho`** (gera `cnn_sapho/cnn_sapho/*.v` + `.mif` com os pesos).
4. Simule (top = `cnn_sapho_top.v`, testbench = `cnn_sapho_test.v`)
   → gera `cnn_sapho/test_output_float.txt`.
5. `python compare.py` → deve dar **PASS** (erro ~1e-4 vs a CNN treinada).

## Notas
- O I/O do SAPHO é inteiro (`fin`=int→float, `fout`=float→int); por isso entrada e saída
  usam **escala 2²⁰** (definida no `.cmm` e no testbench). O `compare.py`/testbench já descalam.
- Os pesos são fixos, derivados da CNN treinada (BatchNorm fundida em afim por canal,
  valores ×escala). Você **não** precisa do repositório `main` nem do `.npz` para rodar.
- `test_output_*`, a forma de onda `*.vcd` e as cópias de `.mif` na raiz são recriados na
  simulação (estão no `.gitignore`).
