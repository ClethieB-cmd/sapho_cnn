// New Verilog file
module cnn_sapho_top (
    input  wire clk, input wire rst,
    input  wire signed [31:0] in,
    output wire req_in,
    output wire signed [31:0] out,
    output wire out_en
);
    cnn_sapho u_cnn (.clk(clk), .rst(rst), .in(in), .out(out), .req_in(req_in), .out_en(out_en));
endmodule
