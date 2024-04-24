yosys read_verilog $::env(VLOG_FILE_NAME)
yosys synth -top $::env(TOP_MODULE) 
yosys flatten
yosys write_verilog netlist/netlist.v
yosys write_json netlist/netlist.json
yosys stat -json