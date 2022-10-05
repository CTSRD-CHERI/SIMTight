#**************************************************************
# This .sdc file is created by Terasic Tool.
# Users are recommended to modify this file to match users logic.
#**************************************************************

#**************************************************************
# Create Clock
#**************************************************************
# CLOCK
create_clock -period 10 [get_ports CLK_100_B3I]
create_clock -period 20 [get_ports CLK_50_B2C]
create_clock -period 20 [get_ports CLK_50_B2L]
create_clock -period 20 [get_ports CLK_50_B3C]
create_clock -period 20 [get_ports CLK_50_B3I]
create_clock -period 20 [get_ports CLK_50_B3L]

create_clock -period "266.666666 MHz" [get_ports DDR4A_REFCLK_p]
create_clock -period "166.666666 MHz" [get_ports DDR4B_REFCLK_p]
create_clock -period "166.666666 MHz" [get_ports DDR4C_REFCLK_p]
create_clock -period "166.666666 MHz" [get_ports DDR4D_REFCLK_p]

#**************************************************************
# Create Generated Clock
#**************************************************************
derive_pll_clocks


#**************************************************************
# Set Clock Latency
#**************************************************************



#**************************************************************
# Set Clock Uncertainty
#**************************************************************
derive_clock_uncertainty


#**************************************************************
# Set Input Delay
#**************************************************************



#**************************************************************
# Set Output Delay
#**************************************************************



#**************************************************************
# Set Clock Groups
#**************************************************************
set_clock_groups -asynchronous -group {[get_clocks { CLK_50_B3I }]}
set_clock_groups -asynchronous -group {[get_clocks { DDR4A_REFCLK_p }]}
set_clock_groups -asynchronous -group {[get_clocks { DDR4B_REFCLK_p }]}
set_clock_groups -asynchronous -group {[get_clocks { DDR4C_REFCLK_p }]}
set_clock_groups -asynchronous -group {[get_clocks { DDR4D_REFCLK_p }]}



#**************************************************************
# Set False Path
#**************************************************************
set_false_path -from * -to [get_cells {DE10_Pro_QSYS_inst|ddr4_status|pio_0|readdata*} ]
set_false_path -from [get_ports CPU_RESET_n*] -to *
set_false_path -from [get_ports BUTTON*] -to *



#**************************************************************
# Set Multicycle Path
#**************************************************************



#**************************************************************
# Set Maximum Delay
#**************************************************************



#**************************************************************
# Set Minimum Delay
#**************************************************************



#**************************************************************
# Set Input Transition
#**************************************************************



#**************************************************************
# Set Load
#**************************************************************



