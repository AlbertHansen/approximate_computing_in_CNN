#%%
import json
import networkx as nx
import os
import matplotlib.pyplot as plt       # for making diagram of graph 
import matplotlib.colors
import pygraphviz as pgv
from networkx.drawing.nx_agraph import graphviz_layout

import warnings
if 'SUPPRESS_FIGURES' in os.environ:
    matplotlib.use('Agg')
    warnings.filterwarnings("ignore", category=UserWarning)

#%% Functions

def add_true_false(graph, driver_list):
    """
    Adds '0' and '1' nodes to the graph and updates the driver list to associate '0' and '1' with the nodes.

    Parameters:
    - graph (Graph): The graph to which the nodes will be added.
    - driver_list (dict): The dictionary containing the driver information.

    Returns:
    None
    """
    graph.add_node('0')
    graph.add_node('1')
    driver_list['0'] = '0'
    driver_list['1'] = '1'

# Add ports as nodes and associate driven nets with them
def add_ports(graph, driver_list, json_netlist, top_module):
    """
    Add ports to the graph and update the driver list and associates driven nets with the ports.

    Args:
        graph (Graph): The graph object to add the ports to.
        driver_list (dict): The dictionary to update with the driver information.
        json_netlist (dict): The JSON netlist containing the module information.
        top_module (str): The name of the top module.

    Returns:
        None
    """
    for port_name, port_data in json_netlist['modules'][top_module]['ports'].items():
        graph.add_node(port_name)

        if port_data['direction'] != 'input':
            continue

        for net in port_data['bits']:
            driver_list[net] = port_name


# add ports as nodes and associate input port connections with nodes
def add_cells(graph, driver_list, json_netlist, top_module):
    """
    Add cells to the graph and update the driver list based on the JSON netlist. The driver list associates the driven nets with the cells.

    Parameters:
    graph (Graph): The graph object to which the cells will be added.
    driver_list (dict): The dictionary to store the driver information.
    json_netlist (dict): The JSON netlist containing the module and cell information.
    top_module (str): The name of the top module.

    Returns:
    None
    """
    for cell_name, cell_data in json_netlist['modules'][top_module]['cells'].items():
        graph.add_node(cell_name)

        outputs = []
        for name, direction in cell_data['port_directions'].items():
            if direction == 'output':
                outputs.append(name)

        for name, connections in cell_data['connections'].items():
            for net in connections:
                if name not in outputs:
                    continue

                driver_list[net] = cell_name

# add edges, ports
def add_port_edges(graph, driver_list, json_netlist, top_module):
    """
    Adds edges to the graph; driving all output ports.

    Args:
        graph (Graph): The graph to which the edges will be added.
        driver_list (dict): A dictionary mapping nets to their corresponding driver nodes.
        json_netlist (dict): The JSON netlist containing module and port information.
        top_module (str): The name of the top module.

    Returns:
        None
    """
    for port_name, port_data in json_netlist['modules'][top_module]['ports'].items():
        
        if port_data['direction'] != 'output':
            continue

        for net in port_data['bits']:
            if net in driver_list:
                graph.add_edge(driver_list[net], port_name)

# add edges, cells
def add_cell_edges(graph, driver_list, json_netlist, top_module):
    """
    Add edges to the graph based on the connections between cells in the netlist; driving all the cells.

    Args:
        graph (Graph): The graph object to which the edges will be added.
        driver_list (list): A list of driver names corresponding to the nets in the netlist.
        json_netlist (dict): The JSON representation of the netlist.
        top_module (str): The name of the top module in the netlist.

    Returns:
        None
    """
    for cell_name, cell_data in json_netlist['modules'][top_module]['cells'].items():
        inputs = []
        for name, direction in cell_data['port_directions'].items():
            if direction == 'input':
                inputs.append(name)

        for name, connections in cell_data['connections'].items():
            for net in connections:
                if name not in inputs:
                    continue

                graph.add_edge(driver_list[net], cell_name)   

# Check for acyclic graph
def check_acyclic(graph):
    """
    Check if a directed graph is acyclic.

    Parameters:
        graph (networkx.DiGraph): The directed graph to be checked.

    Raises:
        Exception: If the graph is not acyclic.

    Returns:
        None
    """
    if not nx.is_directed_acyclic_graph(graph):
        print("Cycles:")
        for cycle in list(nx.simple_cycles(graph)):
            print(f"\t{cycle}")
        raise Exception('The graph is not acyclic!')

#draw graph

def longest_path(graph):
    """
    Finds the longest path in a directed acyclic graph.

    Args:
        graph (nx.DiGraph): The directed acyclic graph.

    Returns:
        Number of nodes in the longest path, inclusive starting and ending points.

    """
    longest_path = nx.dag_longest_path(graph)
    length = len(longest_path) - 2

    print(f"The longest path passes {length} gates:")
    for node in longest_path:
        print(f"\t{node}")

    return len(longest_path)

#%%
def main():
    # Get the top module
    top_module = os.environ.get('TOP_MODULE') # The top design to analyze

    # Load the JSON data
    with open('netlist/netlist.json') as f:
        data = json.load(f)

    # Create a directed graph
    G = nx.DiGraph()
    drivers = {}

    # graph network setup
    add_true_false(G, drivers)
    add_ports(G, drivers, data, top_module)
    add_cells(G, drivers, data, top_module)
    add_port_edges(G, drivers, data, top_module)
    add_cell_edges(G, drivers, data, top_module)

    # Check for cycles
    check_acyclic(G)

    length_of_longest_path = longest_path(G)

    # Draw the graph
    layout = graphviz_layout(G, prog='dot')
    nx.draw(G, layout, arrows=True, node_size=100)
    plt.show()
    plt.savefig('figures/graph.pdf', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()