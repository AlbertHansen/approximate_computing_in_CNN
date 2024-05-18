VLOG_FILE_NAME=./mul8s_1L12/mul8s_1L12.v
C_FILE_NAME=./mul8s_1KV8/mul8s_1KV8.c
TOP_MODULE=mul8s_1L12
METRICS=ER,Gate_Count,CP_Gates
COMPARISON_FILES=mul8s_1KV9,mul8s_1L2J,mul8s_1L12

#Map to the specified list of gate types. Supported gates types are:
#           AND, NAND, OR, NOR, XOR, XNOR, ANDNOT, ORNOT, MUX,
#           NMUX, AOI3, OAI3, AOI4, OAI4. (NOT will always added)
GATES=OR
#GATES=NAND

SUPPRESS_FIGURES=1

# will run regardless of changes in the scripts
.PHONY: netlist
.PHONY: svg
.PHONY: report_gates
.PHONY: report_paths
.PHONY: analysis

# Produces netlist and a file with information w.r.t. the netlist, i.e. gates-counts: output.v and yosys_output.txt, respectively
netlist:
	GATES=$(GATES) \
	VLOG_FILE_NAME=$(VLOG_FILE_NAME) \
	TOP_MODULE=$(TOP_MODULE) \
	yosys netlist.tcl 

svg:
	netlistsvg netlist/netlist.json -o figures/netlist.svg

error_file:
	g++ -std=c++17 -o ./Error/error_executable ./Error/SingleGateEvaluator.cpp ./Error/Evaluator.cpp $(C_FILE_NAME)

report_gates:
	TOP_MODULE=$(TOP_MODULE) \
	SUPPRESS_FIGURES=$(SUPPRESS_FIGURES) \
	python3 gates.py

report_paths:
	TOP_MODULE=$(TOP_MODULE) \
	SUPPRESS_FIGURES=$(SUPPRESS_FIGURES) \
	python3 paths.py

report_error:
	./Error/error_executable
	
histogram:
	SUPPRESS_FIGURES=$(SUPPRESS_FIGURES) \
	python3 ./Error/make_histogram.py

approximate_tripod:
	METRICS=$(METRICS) \
	COMPARISON_FILES=$(COMPARISON_FILES) \
	python3 ./Error/approximation_tripod.py

analysis: netlist svg error_file report_gates report_paths report_error histogram
#analysis: netlist svg report_gates report_paths 