#!/usr/bin/env python3
# Compara a saida do SAPHO com o gabarito da CNN treinada (main).
# Portatil: usa caminhos relativos a este arquivo. Rode com:  python compare.py
import os

D = os.path.dirname(os.path.abspath(__file__))
SAP = os.path.join(D, "cnn_sapho", "test_output_float.txt")   # saida do SAPHO (sim)
EXP = os.path.join(D, "cnn_sapho", "test_expected.txt")       # gabarito (CNN Keras)

def load(p):
    return [float(x) for x in open(p) if x.strip() != ""]

if not os.path.exists(SAP):
    print(f"Nao achei {SAP}\n>>> Rode a simulacao no Aurora primeiro (gera test_output_float.txt).")
    raise SystemExit(1)

sap, exp = load(SAP), load(EXP)
n = min(len(sap), len(exp))
print(f"{'#':>3} {'SAPHO':>16} {'CNN (main)':>16} {'err_abs':>10}")
print("-" * 52)
mae = mx = 0.0; fails = 0
for i in range(n):
    e = abs(sap[i] - exp[i]); mae += e; mx = max(mx, e)
    rel = e / (abs(exp[i]) + 1e-12)
    if e > 1e-3 and rel > 1e-2: fails += 1
    print(f"{i:>3} {sap[i]:>16.10f} {exp[i]:>16.10f} {e:>10.2e}")
print("-" * 52)
print(f"MAE={mae/n:.2e}   MAX={mx:.2e}   ({n} amostras)")
print(">>> PASS  (SAPHO reproduz a CNN treinada)" if fails == 0
      else f">>> FAIL  ({fails}/{n} fora da tolerancia abs<=1e-3 / rel<=1e-2)")
