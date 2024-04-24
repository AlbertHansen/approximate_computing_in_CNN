#%%
import json
import networkx as nx
import matplotlib.pyplot as plt

#%%
# Load the JSON data
with open('netlist/netlist.json') as f:
    data = json.load(f)

# Associate bit/connection names with cell names
cell_names = {}



# Create a directed graph
G = nx.DiGraph()

# Add nodes and edges to the graph
for cell_name, cell_data in data['modules']['mul8s_1KVA']['cells'].items():
    G.add_node(cell_name)
    for i, connection in enumerate(cell_data['connections'].values()):
        for j, node in enumerate(connection):
            print(node)
            G.add_edge(f"cell_{i}_{j}", node)

#draw graph
nx.draw(G)

'''
# Get the input and output ports
input_ports = data['modules']['mul8s_1KVA']['ports']['A']['bits']
output_ports = data['modules']['mul8s_1KVA']['ports']['O']['bits']

# Find the longest path from each input port to each output port
longest_paths = []
for input_port in input_ports:
    for output_port in output_ports:
        if nx.has_path(G, input_port, output_port):
            longest_path = nx.dag_longest_path(G, input_port, output_port)
            longest_paths.append(longest_path)

# Find the overall longest path
overall_longest_path = max(longest_paths, key=len)

# Print the longest path
print('The longest path is:', ' -> '.join(map(str, overall_longest_path)))
'''