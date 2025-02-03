module DE10_Pro(
  input CLK_100_B3I,
  input CLK_50_B2C,
  input CLK_50_B2L,
  input CLK_50_B3C,
  input CLK_50_B3I,
  input CLK_50_B3L,

  input CPU_RESET_n,
  input [1:0] BUTTON,
  input [1:0] SW,
  output [3:0] LED,

  inout SI5340A0_I2C_SCL,
  inout SI5340A0_I2C_SDA,
  input SI5340A0_INTR,
  output SI5340A0_OE_n,
  output SI5340A0_RST_n,

  inout SI5340A1_I2C_SCL,
  inout SI5340A1_I2C_SDA,
  input SI5340A1_INTR,
  output SI5340A1_OE_n,
  output SI5340A1_RST_n,

  output FLASH_CLK,
  output [27:1] FLASH_A,
  inout [15:0] FLASH_D,
  output FLASH_CE_n,
  output FLASH_WE_n,
  output FLASH_OE_n,
  output FLASH_ADV_n,
  output FLASH_RESET_n,
  input FLASH_RDY_BSY_n,

  input  DDR4B_REFCLK_p,
  output [16:0] DDR4B_A,
  output [1:0] DDR4B_BA,
  output [1:0] DDR4B_BG,
  output DDR4B_CK,
  output DDR4B_CK_n,
  output DDR4B_CKE,
  inout  [8:0] DDR4B_DQS,
  inout  [8:0] DDR4B_DQS_n,
  inout  [71:0] DDR4B_DQ,
  inout  [8:0] DDR4B_DBI_n,
  output DDR4B_CS_n,
  output DDR4B_RESET_n,
  output DDR4B_ODT,
  output DDR4B_PAR,
  input DDR4B_ALERT_n,
  output DDR4B_ACT_n,
  input DDR4B_EVENT_n,
  inout DDR4B_SCL,
  inout DDR4B_SDA,
  input DDR4B_RZQ,

  inout FAN_I2C_SCL,
  inout FAN_I2C_SDA,
  inout TEMP_I2C_SCL,
  inout TEMP_I2C_SDA,

  input EXP_EN,

  inout UFL_CLKIN_p,
  inout UFL_CLKIN_n
);

wire reset_n;
wire fan_reset_n;
wire ddr4_local_reset_req;

wire ddr4_b_local_reset_done;
wire ddr4_b_status_local_cal_fail;
wire ddr4_b_status_local_cal_success;

wire [11:0] ddr4_status;

// Reset release
wire ninit_done;
reset_release reset_release (
        .ninit_done(ninit_done)
        );

assign reset_n = !ninit_done && CPU_RESET_n;
assign ddr4_status =
  {ddr4_b_status_local_cal_fail,
     ddr4_b_status_local_cal_success,
       ddr4_b_local_reset_done};

DE10_Pro_QSYS DE10_Pro_QSYS_inst (
        .clk_clk(CLK_50_B3I),
        .reset_reset(~reset_n),
        .emif_s10_ddr4_b_mem_mem_ck(DDR4B_CK),                          
        .emif_s10_ddr4_b_mem_mem_ck_n(DDR4B_CK_n),                        
        .emif_s10_ddr4_b_mem_mem_a(DDR4B_A),                           
        .emif_s10_ddr4_b_mem_mem_act_n(DDR4B_ACT_n),                       
        .emif_s10_ddr4_b_mem_mem_ba(DDR4B_BA),                          
        .emif_s10_ddr4_b_mem_mem_bg(DDR4B_BG),                          
        .emif_s10_ddr4_b_mem_mem_cke(DDR4B_CKE),                         
        .emif_s10_ddr4_b_mem_mem_cs_n(DDR4B_CS_n),                        
        .emif_s10_ddr4_b_mem_mem_odt(DDR4B_ODT),                         
        .emif_s10_ddr4_b_mem_mem_reset_n(DDR4B_RESET_n),                     
        .emif_s10_ddr4_b_mem_mem_par(DDR4B_PAR),                         
        .emif_s10_ddr4_b_mem_mem_alert_n(DDR4B_ALERT_n),                     
        .emif_s10_ddr4_b_mem_mem_dqs(DDR4B_DQS),                         
        .emif_s10_ddr4_b_mem_mem_dqs_n(DDR4B_DQS_n),                       
        .emif_s10_ddr4_b_mem_mem_dq(DDR4B_DQ),                          
        .emif_s10_ddr4_b_mem_mem_dbi_n(DDR4B_DBI_n),                       
        .emif_s10_ddr4_b_oct_oct_rzqin(DDR4B_RZQ),                       
        .emif_s10_ddr4_b_pll_ref_clk_clk(DDR4B_REFCLK_p),                     
        .emif_s10_ddr4_b_status_local_cal_success(ddr4_b_status_local_cal_success),            
        .emif_s10_ddr4_b_status_local_cal_fail(ddr4_b_status_local_cal_fail),   
        .iopll_0_locked_export()
    );

mkDE10FanControl fan_inst(
    .CLK(CLK_50_B2C),
    .RST_N(reset_n),
    .FAN_I2C_SDA(FAN_I2C_SDA),
    .FAN_I2C_SCL(FAN_I2C_SCL),
    .TEMP_I2C_SDA(TEMP_I2C_SDA),
    .TEMP_I2C_SCL(TEMP_I2C_SCL)
);

assign SI5340A0_RST_n = 1'b1;
assign SI5340A1_RST_n = 1'b1;

assign SI5340A0_OE_n = 1'b0;
assign SI5340A1_OE_n = 1'b0;

endmodule
