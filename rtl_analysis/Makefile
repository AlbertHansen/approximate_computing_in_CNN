VLOG_FILE_NAME=./add16se_20J.v
TOP_MODULE=add16se_20J

SUPPRESS_FIGURES=1

# will run regardless of changes in the scripts
.PHONY: netlist
.PHONY: svg
.PHONY: report_gates
.PHONY: report_paths
.PHONY: analysis

# Produces netlist and a file with information w.r.t. the netlist, i.e. gates-counts: output.v and yosys_output.txt, respectively
netlist:
	VLOG_FILE_NAME=$(VLOG_FILE_NAME) \
	TOP_MODULE=$(TOP_MODULE) \
	yosys netlist.tcl 

svg:
	netlistsvg netlist/netlist.json -o figures/netlist.svg

report_gates:
	TOP_MODULE=$(TOP_MODULE) \
	SUPPRESS_FIGURES=$(SUPPRESS_FIGURES) \
	python3 gates.py

report_paths:
	TOP_MODULE=$(TOP_MODULE) \
	SUPPRESS_FIGURES=$(SUPPRESS_FIGURES) \
	python3 paths.py

analysis: netlist svg report_gates report_paths