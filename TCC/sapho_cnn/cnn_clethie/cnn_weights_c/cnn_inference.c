/*
 * cnn_inference.c
 *
 * Inferência CNN em C puro — sem dependências externas.
 * Arquitetura: Conv1D(32,k=5) -> BN -> Conv1D(16,k=3) -> BN -> Flatten -> Dense(64) -> Dense(32) -> Dense(1)
 *
 * Compilar:
 *   gcc -O2 -lm -o cnn_inference cnn_inference.c
 *
 * Para HLS/FPGA: remova #include <math.h> e substitua sqrtf() por implementação
 * própria (ver fast_sqrt abaixo), e remova printf/scanf se necessário.
 */

#include <math.h>   /* sqrtf — remova se for HLS puro */
#include <stdio.h>
#include <string.h>

/* ============================================================
 *  Incluir pesos exportados pelo script Python
 * ============================================================ */
#include "conv1_weights.h"
#include "bn1_weights.h"
#include "conv2_weights.h"
#include "bn2_weights.h"
#include "dense1_weights.h"
#include "dense2_weights.h"
#include "dense3_weights.h"

/* ============================================================
 *  Constantes de arquitetura
 *  (derivadas dos #defines nos headers — não altere aqui)
 * ============================================================ */
#define INPUT_LEN       7          /* N_JANELAMENTO                        */
#define BN_EPS          1e-3f      /* epsilon do BatchNorm do Keras         */

/* Tamanhos dos buffers intermediários */
#define CONV1_OUT_LEN   INPUT_LEN                     /* same padding */
#define CONV2_OUT_LEN   INPUT_LEN                     /* same padding */
#define FLATTEN_SIZE    (CONV2_OUT_LEN * CONV2_FILTERS)  /* 7*16 = 112 */

/* ============================================================
 *  Buffers de ativação (estáticos para evitar stack overflow em FPGA)
 * ============================================================ */
static float buf_conv1[CONV1_OUT_LEN][CONV1_FILTERS];  /* (7, 32) */
static float buf_conv2[CONV2_OUT_LEN][CONV2_FILTERS];  /* (7, 16) */
static float buf_flat [FLATTEN_SIZE];                  /* 112      */
static float buf_d1   [DENSE1_OUT];                    /* 64       */
static float buf_d2   [DENSE2_OUT];                    /* 32       */

/* ============================================================
 *  Utilitários
 * ============================================================ */

/* ReLU escalar */
static inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

/*
 * fast_sqrt — substituto de sqrtf() sem <math.h>, adequado para HLS.
 * Método: Newton-Raphson a partir de estimativa via bit-hack (Quake III).
 * Precisão: ~0.02% após 3 iterações — suficiente para BatchNorm.
 * Comente esta função e use sqrtf() se estiver compilando para CPU normal.
 */
static float fast_sqrt(float x) {
    if (x <= 0.0f) return 0.0f;
    /* Estimativa inicial via bit manipulation */
    union { float f; unsigned int i; } u;
    u.f = x;
    u.i = 0x5f375a86u - (u.i >> 1);   /* rsqrt inicial */
    float y = u.f;
    /* 3 iterações de Newton para rsqrt */
    float half_x = 0.5f * x;
    y = y * (1.5f - half_x * y * y);
    y = y * (1.5f - half_x * y * y);
    y = y * (1.5f - half_x * y * y);
    return x * y;   /* sqrt = x * rsqrt(x) */
}

/* ============================================================
 *  Camada Conv1D com padding='same'
 *
 *  Parâmetros:
 *    input   : ponteiro para float[in_len][in_ch]
 *    in_len  : comprimento da sequência de entrada
 *    in_ch   : número de canais de entrada
 *    kernel  : float[out_ch][ksize][in_ch]
 *    bias    : float[out_ch]
 *    out_ch  : número de filtros de saída
 *    ksize   : tamanho do kernel
 *    output  : ponteiro para float[in_len][out_ch]  (mesmo tamanho — same)
 *    apply_relu : 1 = aplica ReLU, 0 = linear
 * ============================================================ */
static void conv1d_same(
    const float *input,   int in_len, int in_ch,
    const float *kernel,  const float *bias,
    int out_ch, int ksize,
    float *output,
    int apply_relu
) {
    int pad = ksize / 2;   /* padding simétrico para 'same' */

    for (int pos = 0; pos < in_len; pos++) {
        for (int f = 0; f < out_ch; f++) {
            float acc = bias[f];

            for (int k = 0; k < ksize; k++) {
                int src = pos + k - pad;
                if (src < 0 || src >= in_len) continue;   /* zero padding */

                for (int c = 0; c < in_ch; c++) {
                    /* input layout:  [pos][ch]  → acesso: input[src*in_ch + c]  */
                    /* kernel layout: [f][k][c]  → acesso: kernel[(f*ksize + k)*in_ch + c] */
                    acc += input[src * in_ch + c]
                         * kernel[(f * ksize + k) * in_ch + c];
                }
            }

            output[pos * out_ch + f] = apply_relu ? relu(acc) : acc;
        }
    }
}

/* ============================================================
 *  BatchNormalization (modo inferência)
 *
 *  Fórmula:  y = gamma * (x - mean) / sqrt(var + eps) + beta
 *
 *  Parâmetros:
 *    data    : float[len][n_ch]  (in-place)
 *    len     : comprimento da sequência
 *    n_ch    : número de canais
 *    gamma, beta, mean, var : arrays de tamanho n_ch
 * ============================================================ */
static void batch_norm(
    float *data, int len, int n_ch,
    const float *gamma, const float *beta,
    const float *mean,  const float *var
) {
    for (int pos = 0; pos < len; pos++) {
        for (int c = 0; c < n_ch; c++) {
            float x  = data[pos * n_ch + c];
            float std_inv = 1.0f / fast_sqrt(var[c] + BN_EPS);
            data[pos * n_ch + c] = gamma[c] * (x - mean[c]) * std_inv + beta[c];
        }
    }
}

/* ============================================================
 *  Dense (fully-connected)
 *
 *  Parâmetros:
 *    input   : float[in_size]
 *    W       : float[in_size][out_size]  (layout Keras: coluna-major por neurônio)
 *    b       : float[out_size]
 *    output  : float[out_size]
 *    apply_relu : 1 = ReLU, 0 = linear
 * ============================================================ */
static void dense(
    const float *input, int in_size,
    const float *W,     const float *b,
    float *output,      int out_size,
    int apply_relu
) {
    for (int j = 0; j < out_size; j++) {
        float acc = b[j];
        for (int i = 0; i < in_size; i++) {
            /* Keras armazena W como [in][out] */
            acc += input[i] * W[i * out_size + j];
        }
        output[j] = apply_relu ? relu(acc) : acc;
    }
}

/* ============================================================
 *  Função principal de inferência
 *
 *  input  : array de 7 floats (uma janela de amostras)
 *  retorna: amplitude estimada (float)
 * ============================================================ */
float cnn_predict(const float input[INPUT_LEN]) {

    /* ------ Reshape: (7,) -> (7, 1) ------ */
    /* Em C, basta tratar input como input[pos][ch=0].
     * conv1d_same lida com in_ch=1. */

    /* ------ Conv1 ------ */
    conv1d_same(
        input,          INPUT_LEN, /*in_ch=*/ 1,
        &conv1_kernel[0][0][0], conv1_bias,
        CONV1_FILTERS,  CONV1_KERNEL,
        &buf_conv1[0][0],
        /*relu=*/ 1
    );

    /* ------ BatchNorm 1 ------ */
    batch_norm(
        &buf_conv1[0][0], CONV1_OUT_LEN, CONV1_FILTERS,
        bn1_gamma, bn1_beta, bn1_mean, bn1_var
    );

    /* ------ Conv2 ------ */
    conv1d_same(
        &buf_conv1[0][0], CONV1_OUT_LEN, CONV1_FILTERS,
        &conv2_kernel[0][0][0], conv2_bias,
        CONV2_FILTERS,  CONV2_KERNEL,
        &buf_conv2[0][0],
        /*relu=*/ 1
    );

    /* ------ BatchNorm 2 ------ */
    batch_norm(
        &buf_conv2[0][0], CONV2_OUT_LEN, CONV2_FILTERS,
        bn2_gamma, bn2_beta, bn2_mean, bn2_var
    );

    /* ------ Flatten ------ */
    /* buf_conv2 já é contíguo em memória; basta apontar buf_flat */
    memcpy(buf_flat, buf_conv2, FLATTEN_SIZE * sizeof(float));

    /* ------ Dense 1 (64, ReLU) ------ */
    dense(
        buf_flat,     DENSE1_IN,
        &dense1_W[0][0], dense1_b,
        buf_d1,       DENSE1_OUT,
        /*relu=*/ 1
    );

    /* ------ Dense 2 (32, ReLU) ------ */
    dense(
        buf_d1,       DENSE2_IN,
        &dense2_W[0][0], dense2_b,
        buf_d2,       DENSE2_OUT,
        /*relu=*/ 1
    );

    /* ------ Dense 3 / Output (1, linear) ------ */
    float output;
    dense(
        buf_d2,       DENSE3_IN,
        &dense3_W[0][0], dense3_b,
        &output,      DENSE3_OUT,
        /*relu=*/ 0
    );

    return output;
}

/* ============================================================
 *  main — exemplo de uso e teste de sanidade
 *  (remova se for integrar como biblioteca)
 * ============================================================ */
int main(void) {
    printf("=== CNN Inference (C puro) ===\n\n");

    /* Exemplo: amostras com todos zeros */
    float sample_zeros[INPUT_LEN] = {0};
    printf("Input zeros: pred = %.6f\n", cnn_predict(sample_zeros));

    /* Exemplo interativo: leia 7 valores do stdin */
    printf("\nDigite %d valores separados por espaco: ", INPUT_LEN);
    float sample[INPUT_LEN];
    for (int i = 0; i < INPUT_LEN; i++) {
        if (scanf("%f", &sample[i]) != 1) {
            fprintf(stderr, "Erro na leitura.\n");
            return 1;
        }
    }
    float pred = cnn_predict(sample);
    printf("Amplitude estimada: %.6f\n", pred);

    return 0;
}