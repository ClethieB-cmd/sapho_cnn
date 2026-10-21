NOP
#PRNAME cnn_sapho
#NUBITS 32
#NBMANT 23
#NBEXPO 8
#NDSTAC 64
#SDEPTH 16
#NUIOIN 1
#NUIOOU 1
#NUGAIN 128
#FROUND 2
#arrays conv1_W 2 160 "conv1_W.txt"
#arrays conv1_b 2 32 "conv1_b.txt"
#arrays bn1_a 2 32 "bn1_a.txt"
#arrays bn1_b 2 32 "bn1_b.txt"
#arrays conv2_W 2 1536 "conv2_W.txt"
#arrays conv2_b 2 16 "conv2_b.txt"
#arrays bn2_a 2 16 "bn2_a.txt"
#arrays bn2_b 2 16 "bn2_b.txt"
#arrays dense1_W 2 7168 "dense1_W.txt"
#arrays dense1_b 2 64 "dense1_b.txt"
#arrays dense2_W 2 2048 "dense2_W.txt"
#arrays dense2_b 2 32 "dense2_b.txt"
#arrays dense3_W 2 32 "dense3_W.txt"
#arrays dense3_b 2 1 "dense3_b.txt"
#array in_buf 2 7
#array buf_conv1 2 224
#array buf_conv2 2 112
#array buf_flat 2 112
#array buf_d1 2 64
#array buf_d2 2 32
#SHARE jx pred
#SHARE ii pos
JMP main
@conv1_layer LOD 0
SET pos
@__Lwh1 LOD 7
LES pos
JIZ __Lwh1end
LOD 0
SET fx
@__Lwh2 LOD 32
LES fx
JIZ __Lwh2end
LOD fx
LDI conv1_b
SET acc
LOD 0
SET kx
@__Lwh3 LOD 5
LES kx
JIZ __Lwh3end
LOD pos
ADD kx
P_NEG_M 2
S_ADD
SET src
LOD 0
LES src
LIN
P_LOD 7
LES src
S_LAN
JIZ __Lif1else
LOD src
LDI in_buf
P_LOD fx
MLT 5
ADD kx
LDI conv1_W
SF_MLT
SET tmp
LOD acc
F_ADD tmp
SET acc
@__Lif1else LOD kx
ADD 1
SET kx
JMP __Lwh3
@__Lwh3end F_PST_M acc
SET acc
LOD  pos
MLT  32
ADD  fx
P_LOD fx
LDI bn1_a
F_MLT acc
P_LOD fx
LDI bn1_b
SF_ADD
STI buf_conv1
LOD fx
ADD 1
SET fx
JMP __Lwh2
@__Lwh2end LOD pos
ADD 1
SET pos
JMP __Lwh1
@__Lwh1end RET
@conv2_layer LOD 0
SET pos
@__Lwh4 LOD 7
LES pos
JIZ __Lwh4end
LOD 0
SET fx
@__Lwh5 LOD 16
LES fx
JIZ __Lwh5end
LOD fx
LDI conv2_b
SET acc
LOD 0
SET kx
@__Lwh6 LOD 3
LES kx
JIZ __Lwh6end
LOD pos
ADD kx
P_NEG_M 1
S_ADD
SET src
LOD 0
LES src
LIN
P_LOD 7
LES src
S_LAN
JIZ __Lif2else
LOD 0
SET cx
@__Lwh7 LOD 32
LES cx
JIZ __Lwh7end
LOD src
MLT 32
ADD cx
LDI buf_conv1
P_LOD fx
MLT 3
ADD kx
MLT 32
ADD cx
LDI conv2_W
SF_MLT
SET tmp
LOD acc
F_ADD tmp
SET acc
LOD cx
ADD 1
SET cx
JMP __Lwh7
@__Lwh7end @__Lif2else LOD kx
ADD 1
SET kx
JMP __Lwh6
@__Lwh6end F_PST_M acc
SET acc
LOD  pos
MLT  16
ADD  fx
P_LOD fx
LDI bn2_a
F_MLT acc
P_LOD fx
LDI bn2_b
SF_ADD
STI buf_conv2
LOD fx
ADD 1
SET fx
JMP __Lwh5
@__Lwh5end LOD pos
ADD 1
SET pos
JMP __Lwh4
@__Lwh4end RET
@dense_layers LOD 0
SET pos
@__Lwh8 LOD 7
LES pos
JIZ __Lwh8end
LOD 0
SET jx
@__Lwh9 LOD 16
LES jx
JIZ __Lwh9end
LOD pos
MLT 16
ADD jx
P_LOD pos
MLT 16
ADD jx
LDI buf_conv2
STI buf_flat
LOD jx
ADD 1
SET jx
JMP __Lwh9
@__Lwh9end LOD pos
ADD 1
SET pos
JMP __Lwh8
@__Lwh8end LOD 0
SET jx
@__Lwh10 LOD 64
LES jx
JIZ __Lwh10end
LOD jx
LDI dense1_b
SET acc
LOD 0
SET ii
@__Lwh11 LOD 112
LES ii
JIZ __Lwh11end
LOD ii
LDI buf_flat
P_LOD ii
MLT 64
ADD jx
LDI dense1_W
SF_MLT
SET tmp
LOD acc
F_ADD tmp
SET acc
LOD ii
ADD 1
SET ii
JMP __Lwh11
@__Lwh11end LOD jx
PF_PST_M acc
STI buf_d1
LOD jx
ADD 1
SET jx
JMP __Lwh10
@__Lwh10end LOD 0
SET jx
@__Lwh12 LOD 32
LES jx
JIZ __Lwh12end
LOD jx
LDI dense2_b
SET acc
LOD 0
SET ii
@__Lwh13 LOD 64
LES ii
JIZ __Lwh13end
LOD ii
LDI buf_d1
P_LOD ii
MLT 32
ADD jx
LDI dense2_W
SF_MLT
SET tmp
LOD acc
F_ADD tmp
SET acc
LOD ii
ADD 1
SET ii
JMP __Lwh13
@__Lwh13end LOD jx
PF_PST_M acc
STI buf_d2
LOD jx
ADD 1
SET jx
JMP __Lwh12
@__Lwh12end LOD_V dense3_b 0
SET acc
LOD 0
SET ii
@__Lwh14 LOD 32
LES ii
JIZ __Lwh14end
LOD ii
LDI buf_d2
P_LOD ii
LDI dense3_W
SF_MLT
SET tmp
LOD acc
F_ADD tmp
SET acc
LOD ii
ADD 1
SET ii
JMP __Lwh14
@__Lwh14end LOD acc
SET pred
RET
@main @__Lwh15 LOD 0
SET ii
@__Lwh16 LOD 7
LES ii
JIZ __Lwh16end
LOD ii
PF_INN 0
P_LOD 1048576.0
SF_DIV
NOP
F_QUO
STI in_buf
LOD ii
ADD 1
SET ii
JMP __Lwh16
@__Lwh16end CAL conv1_layer
CAL conv2_layer
CAL dense_layers
LOD 1048576.0
F_MLT pred
F2I
OUT 0
JMP __Lwh15
@__Lwh15end @fim JMP fim
