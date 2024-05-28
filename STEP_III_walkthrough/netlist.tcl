yosys read_verilog $::env(VLOG_FILE_NAME)
yosys synth -top $::env(TOP_MODULE) 
yosys flatten
yosys abc -g $::env(GATES)
yosys write_verilog netlist/netlist.v
yosys write_json netlist/netlist.json

# If using abc -g 
#-g type1,type2,...
#        Map to the specified list of gate types. Supported gates types are:
#           AND, NAND, OR, NOR, XOR, XNOR, ANDNOT, ORNOT, MUX,
#           NMUX, AOI3, OAI3, AOI4, OAI4.
#        (The NOT gate is always added to this list automatically.)

#        The following aliases can be used to reference common sets of gate
#        types:
#          simple: AND OR XOR MUX
#          cmos2:  NAND NOR
#          cmos3:  NAND NOR AOI3 OAI3
#          cmos4:  NAND NOR AOI3 OAI3 AOI4 OAI4
#          cmos:   NAND NOR AOI3 OAI3 AOI4 OAI4 NMUX MUX XOR XNOR
#          gates:  AND NAND OR NOR XOR XNOR ANDNOT ORNOT
#          aig:    AND NAND OR NOR ANDNOT ORNOT

#        The alias 'all' represent the full set of all gate types.

#        Prefix a gate type with a '-' to remove it from the list. For example
#        the arguments 'AND,OR,XOR' and 'simple,-MUX' are equivalent.

#        The default is 'all,-NMUX,-AOI3,-OAI3,-AOI4,-OAI4'.