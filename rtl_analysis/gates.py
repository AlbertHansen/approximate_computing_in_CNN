#%%
import json
import os
import matplotlib.pyplot as plt       # for making diagram of graph 
import matplotlib

import warnings
if 'SUPPRESS_FIGURES' in os.environ:
    matplotlib.use('Agg')
    warnings.filterwarnings("ignore", category=UserWarning)

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



#%%

def main():
    top_module = os.environ.get('TOP_MODULE') # The top design to analyze

    with open('netlist/netlist.json') as f:
        data = json.load(f)

    gates = count_gates(data, top_module)
    
    total_gates = sum(gates.values())

    print(f"The total gatecount is {total_gates} gates of the following types:")
    for gate, count in gates.items():
        gate = gate.strip('$_')
        print(f'\t{gate}:\t{count}')

if __name__ == '__main__':
    main()