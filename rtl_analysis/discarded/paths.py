import json
import subprocess
import os

def get_info():
    """
    Retrieves information from the 'netlist.json' file.

    Returns:
        dict: A dictionary containing the information from the 'netlist.json' file.
    """
    with open('netlist/netlist.json') as f:
        info = json.load(f)
    return info

def get_ports(json_info):
    """
    Get the input and output ports from the given JSON information.

    Args:
        json_info (dict): A dictionary containing the JSON information.

    Returns:
        tuple: A tuple containing two lists - input_ports and output_ports.
               input_ports (list): A list of input port names.
               output_ports (list): A list of output port names.
    """
    input_ports = []
    output_ports = []
    top_module = os.environ.get('TOP_MODULE')
    # Iterate over the modules
    for module_key, module_value in json_info["modules"].items():

        # Get the module name
        print(module_key)

        # Verify that the module is the top module
        if module_key != top_module:
            continue

        # Iterate over the ports
        for port_name, port_data in module_value["ports"].items():
            # Check the direction of the port and add it to the appropriate list
            if port_data["direction"] == "input":
                input_ports.append(port_name)
            elif port_data["direction"] == "output":
                output_ports.append(port_name)

    return input_ports, output_ports

def generate_paths(input_ports, output_ports):
    """
    Generate paths between input and output ports and save them in a file.

    Args:
        input_ports (list): A list of input ports.
        output_ports (list): A list of output ports.

    Returns:
        None
    """
    # clean previous paths in file
    subprocess.run('echo " " | tee work/netlist_paths.txt', shell=True)

    # iterate over input and output ports and save paths in netlist_paths.txt
    for input_port in input_ports:
        for output_port in output_ports:
            # Generate the command
            command = f"netlist-paths --from {input_port} --to {output_port} --all-paths --ignore-hierarchy-markers netlist/netlist.xml | tee -a work/netlist_paths.txt"
            print(command)
            # run the command
            # subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import json

def parse_txt_to_json(txt_file):
    """
    Parses a text file containing paths information and converts it into a JSON file.

    Args:
        txt_file (str): The path to the input text file.

    Returns:
        list: A list of dictionaries representing the paths information.

    Example:
        parse_txt_to_json('/path/to/input.txt')
    """
    with open(txt_file, 'r') as f:
        lines = f.readlines()

    paths = []
    path = {}
    keys = ['Name', 'Type', 'DType', 'Statement', 'Location']
    line1_flag = False
    for line in lines:
        if 'Path' in line:
            if path:
                paths.append(path)
            path = {keys[0]: [], keys[1]: [], keys[2]: [], keys[3]: [], keys[4]: []}
            continue
        elif 'Name' in line:      # skip line with keys
            line1_flag = True
            continue
        elif line1_flag:          # skip separator line
            line1_flag = False
            continue
        elif line.strip():
            parts = line.split()
            i = 0
            for part in parts:
                if '[' in part:    # remove bus indications
                    continue
                path[keys[i]].append(part)
                i += 1

    if path:
        paths.append(path)

    # write to json file
    with open('information/netlist_paths.json', 'w') as f:
        json.dump(paths, f, indent=4)

    return paths

def get_longest_path(paths):
    """
    Returns the longest path from a list of paths (JSON-formatted as in 'parse_txt_to_json(txt_file)).

    Args:
        paths (list): A list of paths, where each path is a dictionary with a 'Name' key.

    Returns:
        dict: The longest path from the list of paths.

    """
    longest_path = paths[0]
    for path in paths:
        if len(path['Name']) > len(longest_path['Name']):
            longest_path = path

    return longest_path

def print_longest_path(longest_path):
    """
    Prints the longest path.

    Args:
        longest_path (dict): A dictionary containing the longest path information.

    Returns:
        None
    """
    print("Longest path:")
    print("\tNumber of gates passed: ", len(longest_path['Name']) - 2)
    print("\tPath:", ' '.join(longest_path['Name']))
    print("\tTraceability:", ' '.join(longest_path['Location']) )


def main():
    info = get_info()
    input_ports, output_ports = get_ports(info)
    print("Input ports:", input_ports)
    print("Output ports:", output_ports)
    generate_paths(input_ports, output_ports)
    #paths = parse_txt_to_json('work/netlist_paths.txt')
    #longest_path = get_longest_path(paths)
    #print_longest_path(longest_path)


if __name__ == '__main__':
    main()