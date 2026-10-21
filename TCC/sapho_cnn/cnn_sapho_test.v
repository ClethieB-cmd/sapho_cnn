`timescale 1ns/1ps
// Testbench do processador unico cnn_sapho (Fase B).
// Le test_input_sapho.txt (140 inteiros = 20 janelas x 7) e captura 20 predicoes.
// Saida bruta -> test_output_sapho.txt ; descalada (/2^20) -> test_output_float.txt
//
// >>>>> UNICO AJUSTE NECESSARIO POR MAQUINA <<<<<
// Aponte PROJ para a pasta do processador cnn_sapho NO SEU PC (barras "/").
`define PROJ "C:/Intel/tcc/sapho_cnn-main/cnn_sapho"

module cnn_sapho_test ();
    reg clk = 0, rst = 1;
    initial #20 rst = 0;
    always #5 clk = ~clk;

    reg  signed [31:0] in_data = 0;
    wire signed [31:0] out_data;
    wire req_in, out_en;

    cnn_sapho_top dut (.clk(clk), .rst(rst), .in(in_data),
                       .out(out_data), .req_in(req_in), .out_en(out_en));

    integer fin_fp, fout_int_fp, fout_flt_fp, sc, n;
    reg signed [31:0] sbuf = 0;
    real v_dec;
    initial begin
        n = 0;
        fin_fp      = $fopen({`PROJ, "/test_input_sapho.txt"},  "r");
        fout_int_fp = $fopen({`PROJ, "/test_output_sapho.txt"}, "w");
        fout_flt_fp = $fopen({`PROJ, "/test_output_float.txt"}, "w");
    end

    always @(negedge clk) if (!rst && req_in) begin
        sc = $fscanf(fin_fp, "%d", sbuf);
        if (sc != 1) sbuf <= 0;
    end
    always @(*) in_data = req_in ? sbuf : 0;

    always @(posedge clk) if (out_en) begin
        $fdisplay(fout_int_fp, "%0d", out_data); $fflush(fout_int_fp);
        v_dec = $itor(out_data) / 1048576.0;          // descala (SCALE_OUT = 2^20)
        $fdisplay(fout_flt_fp, "%.10f", v_dec); $fflush(fout_flt_fp);  // 10 casas (igual ao expected)
        n = n + 1;
        $display("[OUT #%0d] raw=%0d  pred=%.10f", n, out_data, v_dec);
        if (n >= 20) begin $display("Info: 20 predicoes. Fim."); $finish; end
    end

    initial begin #200000000; $display("TIMEOUT n=%0d/20", n); $finish; end
endmodule
