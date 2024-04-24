#%%
import json
import os
import matplotlib.pyplot as plt       # for making diagram of graph 
import matplotlib
import csv

import warnings
if 'SUPPRESS_FIGURES' in os.environ:
    matplotlib.use('Agg')
    warnings.filterwarnings("ignore", category=UserWarning)

#%% Transistors required per logic gate:

# Based on edge copilot: 
'''
transistor_multiplier = {
        'NOT'   : 1,    
        'AND'   : 2,    # http://bear.ces.cwru.edu/eecs_318/eecs_318_5.pdf
        'NAND'  : 1,    # https://eepower.com/technical-articles/basic-cmos-logic-gates/
        'OR'    : 1,    # https://eepower.com/technical-articles/basic-cmos-logic-gates/
        'NOR'   : 1,    # https://eepower.com/technical-articles/basic-cmos-logic-gates/
        'XOR'   : 5,    # https://forum.allaboutcircuits.com/threads/number-of-cmos-transistors-required-for.43613/
        'XNOR'  : 5,    # https://www.homemade-circuits.com/how-to-make-logic-gates-using-transistors/
        'ANDNOT': 3,    # based on 2 for and and 1 for not'ing the output 
        'ORNOT' : 2,    # based on 1 for or and 1 for not'ing the output 
        'MUX'   : 8,    # https://electronics.stackexchange.com/questions/141943/determine-the-number-of-transistors-needed-to-build-cmos-circuit
        'NMUX'  : 8,    # OBS: This is defaulted to the same as MUX
        'AOI3'  : 12,   # https://electronics.stackexchange.com/questions/141943/determine-the-number-of-transistors-needed-to-build-cmos-circuit
        'OAI3'  : 12,   # https://electronics.stackexchange.com/questions/141943/determine-the-number-of-transistors-needed-to-build-cmos-circuit
        'AOI4'  : 16,   # https://en.wikipedia.org/wiki/AND-OR-invert
        'OAI4'  : 16    # OBS: This is a quess (based on AOI4)
}
'''

# Based on wikipedia and googling (the lowest count, optimistic)
transistor_multiplier = {
        'NOT'   : 1,    # https://en.wikipedia.org/wiki/Inverter_(logic_gate)
        'AND'   : 2,    # https://en.wikipedia.org/wiki/AND_gate
        'NAND'  : 2,    # https://mavink.com/explore/Nand-Gate-Transistor-Circuit
        'OR'    : 2,    # https://www.homemade-circuits.com/how-to-make-logic-gates-using-transistors/
        'NOR'   : 2,    # https://www.electronics-tutorials.ws/logic/logic_6.html
        'XOR'   : 6,    # https://en.wikipedia.org/wiki/XOR_gate
        'XNOR'  : 8,   # https://en.wikipedia.org/wiki/XNOR_gate
        'ANDNOT': 3,    # AND + NOT  
        'ORNOT' : 3,    # OR + NOT
}

#%% Functions
def count_gates(json_netlist, module):
    """
    Count the number of each gate type in a given module of a JSON netlist.

    Parameters:
    json_netlist (dict): The JSON netlist containing the module data.
    module (str): The name of the module to analyze.

    Returns:
    dict: A dictionary where the keys are the gate types and the values are the counts of each gate type.
    """
    module_data = json_netlist['modules'][module]

    gate_types = {}
    for cell_name, cell_data in module_data['cells'].items():
        del cell_name
        if cell_data['type'] in gate_types:
            gate_types[cell_data['type']] += 1
        else:
            gate_types[cell_data['type']] = 1

    return gate_types

import csv

def save_summary(gates, transistor_count):
    """
    Save the summary of gates and transistor count to a CSV file.

    Args:
        gates (dict): A dictionary containing the count of each gate type.
        transistor_count (int): The total count of transistors.

    Returns:
        None
    """
    summary = gates
    summary['gatecount'] = sum(gates.values())
    summary['transistorcount'] = transistor_count
    summary = {name.strip("$_"): value for (name, value) in summary.items()}

    field_names = list(summary.keys())

    with open('summary/gates.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        writer.writerow(summary)

#%%

def main():
    top_module = os.environ.get('TOP_MODULE') # The top design to analyze

    with open('netlist/netlist.json') as f:
        data = json.load(f)

    gates = count_gates(data, top_module)
    
    # count gates and transistors
    total_gates = sum(gates.values())
    total_transistors = 0
    for gate, count in gates.items():
        gate = gate.strip('$_')
        total_transistors += count * transistor_multiplier[gate]

    '''
    print(f"The total gatecount is {total_gates} gates ({total_transistors} transistors) of the following types:")
    for gate, count in gates.items():
        gate = gate.strip('$_')
        transistors = count * transistor_multiplier[gate]
        print(f'\t{gate}:\t{count}\t({transistors})')
    '''
    save_summary(gates, total_transistors)

if __name__ == '__main__':
    main()