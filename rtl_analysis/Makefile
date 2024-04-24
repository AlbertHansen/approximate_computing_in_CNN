VLOG_FILE_NAME=./mul8s_1KVA.v
TOP_MODULE=mul8s_1KVA

XML_TOOL = netlist-paths
XML_TOOL_FLAGS = --verbose --compile netlist/netlist.v

# 
.PHONY: netlist
.PHONY: svg
.PHONY: xml
.PHONY: report_gates
.PHONY: report_paths
.PHONY: analysis


# Produces netlist and a file with information w.r.t. the netlist, i.e. gates-counts: output.v and yosys_output.txt, respectively
netlist:
	VLOG_FILE_NAME=$(VLOG_FILE_NAME) \
	TOP_MODULE=$(TOP_MODULE) \
	yosys netlist.tcl 
#| tee ./work/netlist_information.txt

svg:
	netlistsvg netlist/netlist.json -o figures/netlist.svg

xml:
	$(XML_TOOL) $(XML_TOOL_FLAGS) -o ./netlist/netlist.xml

report_gates:
	python3 gates.py

report_paths:
	TOP_MODULE=$(TOP_MODULE) \
	python3 paths.py

analysis: netlist svg xml report_gates report_paths