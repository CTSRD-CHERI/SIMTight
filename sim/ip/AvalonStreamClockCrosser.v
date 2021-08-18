module AvalonStreamClockCrosser #(
		parameter DATA_WIDTH = 8
	) (
		input  wire in_clk,
		input  wire in_reset,
		input  wire out_clk,
		input  wire out_reset,
		output wire in_ready,
		input  wire in_valid,
		input  wire [DATA_WIDTH-1:0] in_data,
		input  wire out_ready,
		output wire out_valid,
		output wire [DATA_WIDTH-1:0] out_data
	);

  assign out_data = in_data;
  assign out_valid = in_valid;
  assign in_ready = out_ready;

endmodule
