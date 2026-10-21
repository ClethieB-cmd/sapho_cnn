# TCC: CNN treinada rodando no processador SAPHO

Uma rede neural convolucional (CNN) treinada para estimar amplitude de pulsos, traduzida
para C± e executada como programa no SAPHO, o processador soft-core do NIPS-CERN (UFJF)
cuja arquitetura é gerada sob demanda a partir do programa.

## Conteúdo

| Pasta | O que é |
|---|---|
| [`sapho_cnn/`](sapho_cnn/) | Projeto AURORA: a CNN em C± (`cnn_sapho/cnn_sapho.cmm`), os pesos treinados, o processador gerado, o testbench e o script de comparação. Como rodar: [`sapho_cnn/README.md`](sapho_cnn/README.md) |
| [`sapho_cnn/cnn_clethie/`](sapho_cnn/cnn_clethie/) | Treinamento da CNN em Python, exportação dos pesos e a inferência de referência em C |
| [`../procTest_00/`](../procTest_00/) | Versão anterior da CNN em C±, na AURORA 4.1 (projeto `leandro.spf`, na raiz do repositório) |

## Origem e créditos

- O projeto `sapho_cnn` parte do repositório
  [Chrysthofer/sapho_cnn](https://github.com/Chrysthofer/sapho_cnn), de
  Chrysthofer Arthur, na versão de 11/07/2026 (commit `87dc6da`). O
  `sapho_cnn/README.md` é o dele, com os caminhos atualizados para a estrutura atual.
- O treinamento e a exportação em `cnn_clethie/` são de Diogo Cardinot, como diz o
  README daquela pasta.

## O que mudou em relação à origem

- `cnn_sapho/cnn_sapho.cmm`: diretiva `#FROUND 2` incluída e `#PRNAME cnn_sapho` retirada.
- `cnn_clethie/trainingCNN.py` e `export_weights.py`: o caminho do dataset passou a ser
  relativo à pasta do script, em vez de fixo no PC do autor, e pode ser passado como argumento.
- `cnn_sapho_test.v`: `PROJ` aponta para `C:/Intel/tcc/sapho_cnn-main/cnn_sapho`. Ajuste
  para a pasta onde o projeto estiver.
- `testbench/*.json`: caminhos absolutos do PC do autor trocados por relativos.
- `cnn_sapho/test_expected.txt`: mantido aqui. Saiu do repositório de origem no último
  commit, mas o `compare.py` precisa dele.
- Os arquivos que na origem ficavam em `cnn_sapho/Software/` estão em `cnn_sapho/`, e o
  processador gerado fica em `cnn_sapho/cnn_sapho/`.

## Último resultado

Simulação de 19/10/2026 (AURORA 6.3.2, Verilator), 20 janelas de teste de 7 amostras:
`python compare.py` dá **PASS**, com erro absoluto médio de 5,15e-06 e máximo de 1,43e-05
entre a saída do SAPHO e a da CNN em Keras.

Esse número mede a fidelidade da portação (o SAPHO reproduz a rede de referência). Não
mede a acurácia da CNN na estimação de amplitude, que é outra pergunta.

## O que não está no repositório

| Item | Por quê |
|---|---|
| `.aurora/`, `.slang/` | Cache e configuração local da AURORA, com caminhos deste PC |
| `cnn_sapho_test.vcd` | Forma de onda da simulação, 17 MB, recriada a cada simulação |
| `cnn_sapho/test_output_*.txt` | Saída da simulação, recriada a cada simulação |
| `.mif` e `pc_cnn_sapho_mem.txt` na raiz de `sapho_cnn/` | Cópias idênticas das que estão em `cnn_sapho/cnn_sapho/` |
| `cnn_inference.exe` | Binário; recompile com `gcc -O2 -lm -o cnn_inference cnn_inference.c` |
| `ManipulacaoDados/` (dataset `.npz`) | Só é preciso para treinar de novo; não é preciso para rodar no SAPHO |
