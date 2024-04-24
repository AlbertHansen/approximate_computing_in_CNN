import json
import re

def extract_json_from_txt(txt_file, json_file):
    """
    Extracts a JSON string from a text file and writes it to a JSON file.

    Args:
        txt_file (str): The path to the input text file.
        json_file (str): The path to the output JSON file.

    Returns:
        None

    Raises:
        None

    """
    # Open the text file
    with open(txt_file, 'r') as f:
        # Read the content of the file
        content = f.read()

    # Use regex to find the JSON formatted string
    json_str_match = re.search(r'\{.*\}', content, re.DOTALL)

    # Check if a JSON string was found
    if json_str_match is None:
        print("No JSON string found in the text file.")
        return

    # Get the JSON string
    json_str = json_str_match.group()

    # Try to load the JSON string into a Python dictionary
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        print("The JSON string in the text file is not well-formed.")
        return

    # Write the dictionary to a JSON file
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)

    return data

def get_info(data):
    """
    Prints the total number of gates and the count of each gate type.

    Args:
        data (dict): A dictionary containing the design information.

    Returns:
        None
    """
    design = data['design']
    gates = design['num_cells_by_type']
    gatecount = 0
    gatecount_per_type = []
    for key, value in gates.items():
        key = key.strip("_$")
        gatecount += value
        gatecount_per_type.append(f"{key}:\t{value}")
    
    print(f"Total number of gates: {gatecount}")
    for gate in gatecount_per_type:
        print(f"\t{gate}")


def main():
    data = extract_json_from_txt("work/netlist_information.txt", "information/gates.json")
    get_info(data)

if __name__ == '__main__':
    main()