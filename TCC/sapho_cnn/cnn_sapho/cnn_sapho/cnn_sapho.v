module cnn_sapho (

input  clk, rst,
input  signed [31:0] in ,
output signed [31:0] out,
output [0:0] req_in,
output [0:0] out_en);

/* verilator tracing_off */

wire itr = 1'b0;
wire cheguei;
wire [0:0] lk_toma, lk_cade, lk_valeu, lk_taqui;
wire [31:0] lk_cade_dado;
wire [31:0] lk_toma_dado;
assign lk_valeu[0] = 1'b0;
assign lk_taqui[0] = 1'b0;
assign lk_cade_dado[31:0] = 0;
wire proc_req_in, proc_out_en;
wire [0:0] addr_in;
wire [0:0] addr_out;

`ifdef __ICARUS__
 `ifndef YANC_SIM_VIS
  `define YANC_SIM_VIS
 `endif
`endif
`ifdef YANC_TRACE
 `ifndef YANC_SIM_VIS
  `define YANC_SIM_VIS
 `endif
`endif

`ifdef YANC_SIM_VIS
wire mem_wr;
wire [13:0] mem_addr_wr;
wire [8:0] pc_sim_val;
`endif

processor#(.NUBITS(32),
.NBMANT(23),
.NBEXPO(8),
.NBOPER(14),
.NUGAIN(128),
.MDATAS(11755),
.MINSTS(291),
.SDEPTH(16),
.DDEPTH(64),
.NBIOIN(1),
.NBIOOU(1),
.FFTSIZ(3),
.FROUND(2),
.LOD(1),
.SET(1),
.LES(1),
.JIZ(1),
.LDI(1),
.ADD(1),
.P_NEG_M(1),
.S_ADD(1),
.LIN(1),
.P_LOD(1),
.S_LAN(1),
.MLT(1),
.SF_MLT(1),
.F_ADD(1),
.F_PST_M(1),
.F_MLT(1),
.SF_ADD(1),
.STI(1),
.PF_PST_M(1),
.PF_INN(1),
.SF_DIV(1),
.F_DIV(1),
.CAL(1),
.F2I(1),
.OUT(1),
.DFILE("cnn_sapho_data.mif"),
.IFILE("cnn_sapho_inst.mif"))

`ifdef YANC_SIM_VIS
p_cnn_sapho (clk, rst, in, out, addr_in, addr_out, proc_req_in, proc_out_en, itr, cheguei, lk_toma, lk_cade, lk_valeu, lk_taqui, lk_cade_dado, lk_toma_dado, mem_wr, mem_addr_wr,pc_sim_val);
`else
p_cnn_sapho (clk, rst, in, out, addr_in, addr_out, proc_req_in, proc_out_en, itr, cheguei, lk_toma, lk_cade, lk_valeu, lk_taqui, lk_cade_dado, lk_toma_dado);
`endif

assign req_in = proc_req_in;
assign out_en = proc_out_en;

/* verilator tracing_on */

// ----------------------------------------------------------------------------
// Simulation -----------------------------------------------------------------
// ----------------------------------------------------------------------------

`ifdef YANC_SIM_VIS

// I/O ------------------------------------------------------------------------

reg signed [31:0] in_sim_0 = 0;
reg req_in_sim_0 = 0;

reg signed [31:0] out_sig_0 = 0;
reg out_en_sim_0 = 0;

always @ (posedge clk) begin
   if (req_in == 1) in_sim_0 <= in;
end
always @ (*) begin
   req_in_sim_0 = req_in == 1;
end

always @ (posedge clk) begin
   if (out_en == 1) out_sig_0 <= out;
end
always @ (*) begin
   out_en_sim_0 = out_en == 1;
end

// variables ------------------------------------------------------------------

/* verilator tracing_off */  // float decode helpers (not traced)
reg signed [23:0] sm_me2; always @ (*) sm_me2 = (out[31]) ? -$signed({1'b0, out[22:0]}) : $signed({1'b0, out[22:0]});
reg signed [7:0] e_me2; always @ (*)  e_me2 = $signed(out[30:23]);
/* verilator tracing_on */

reg [31:0] me1_f_global_v_pos_e_ = 0;
reg [31:0] me1_f_global_v_fx_e_ = 0;
real me2_f_global_v_acc_e_ = 0.0;
reg [31:0] me1_f_global_v_kx_e_ = 0;
reg [31:0] me1_f_global_v_src_e_ = 0;
real me2_f_global_v_tmp_e_ = 0.0;
reg [31:0] me1_f_global_v_cx_e_ = 0;
real me2_f_global_v_pred_e_ = 0.0;
reg [31:0] me1_f_global_v_jx_e_ = 0;
reg [31:0] me1_f_global_v_ii_e_ = 0;

always @ (posedge clk) begin
   if (mem_wr && (pc_sim_val == 4 || pc_sim_val == 66 || pc_sim_val == 70 || pc_sim_val == 145 || pc_sim_val == 149 || pc_sim_val == 172)) me1_f_global_v_pos_e_ <= out;
   if (mem_addr_wr == 11739 && mem_wr) me1_f_global_v_fx_e_ <= out;
   if (mem_addr_wr == 11741 && mem_wr) me2_f_global_v_acc_e_ <= sm_me2*$pow(2.0,e_me2);
   if (mem_addr_wr == 11742 && mem_wr) me1_f_global_v_kx_e_ <= out;
   if (mem_addr_wr == 11745 && mem_wr) me1_f_global_v_src_e_ <= out;
   if (mem_addr_wr == 11746 && mem_wr) me2_f_global_v_tmp_e_ <= sm_me2*$pow(2.0,e_me2);
   if (mem_addr_wr == 11750 && mem_wr) me1_f_global_v_cx_e_ <= out;
   if (mem_wr && (pc_sim_val == 265)) me2_f_global_v_pred_e_ <= sm_me2*$pow(2.0,e_me2);
   if (mem_wr && (pc_sim_val == 154 || pc_sim_val == 168 || pc_sim_val == 175 || pc_sim_val == 207 || pc_sim_val == 210 || pc_sim_val == 242)) me1_f_global_v_jx_e_ <= out;
   if (mem_wr && (pc_sim_val == 183 || pc_sim_val == 200 || pc_sim_val == 218 || pc_sim_val == 235 || pc_sim_val == 247 || pc_sim_val == 262 || pc_sim_val == 268 || pc_sim_val == 281)) me1_f_global_v_ii_e_ <= out;
end

// instructions ---------------------------------------------------------------

reg [31:0] valr2=0;
/* verilator tracing_off */
reg [31:0] valr1 /* verilator public_flat */=0;
reg [31:0] valr3 /* verilator public_flat */=0;
reg [31:0] valr4 /* verilator public_flat */=0;
reg [31:0] valr5 /* verilator public_flat */=0;
reg [31:0] valr6 /* verilator public_flat */=0;
reg [31:0] valr7 /* verilator public_flat */=0;
reg [31:0] valr8 /* verilator public_flat */=0;
reg [31:0] valr9 /* verilator public_flat */=0;
reg [31:0] valr10 /* verilator public_flat */=0;
/* verilator tracing_on */

reg [19:0] min [0:291-1];

/* verilator tracing_off */ reg signed [19:0] linetab /* verilator public_flat */ =-1; /* verilator tracing_on */
reg signed [19:0] linetabs=-1;

initial	$readmemb("pc_cnn_sapho_mem.txt",min);

always @ (posedge clk) begin
if (pc_sim_val < 291) linetab <= min[pc_sim_val];
linetabs <= linetab;   
valr1    <= {{(23){1'b0}}, pc_sim_val};
valr2    <= valr1;
valr3    <= valr2;
valr4    <= valr3;
valr5    <= valr4;
valr6    <= valr5;
valr7    <= valr6;
valr8    <= valr7;
valr9    <= valr8;
valr10   <= valr9;
end

`endif

endmodule