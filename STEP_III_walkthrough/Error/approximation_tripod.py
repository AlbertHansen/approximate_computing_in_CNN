import os
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_csv_value(filename, row_number, column_number):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        value = rows[row_number][column_number]
        return float(value)

def read_values_from_file(filename):
    try:
        with open(filename, 'r') as file:
            for line in file:
                # Split each line based on tabs
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    key, value = parts
                    if key == "Number of Gates:":
                        gates = int(value)
                    elif key == "Total Delay:":
                        delay = float(value.split()[0])  # Extract the numeric part only
        return gates, delay
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None, None
    except Exception as e:
        print(f"Error reading file '{filename}': {e}")
        return None, None

def check_string_count(strings_list):
    if len(strings_list) != 3:
        
        raise ValueError("The list must contain exactly 3 strings")

class MetricsStruct:
    def __init__(self,name,value):
        self.name = name
        self.value = value 


def main():
    metrics = os.environ.get('METRICS')
    metrics = list(metrics.split(","))
    comparison_files = os.environ.get('COMPARISON_FILES')
    comparison_files = list(comparison_files.split(","))

    cost_results = {}
    performance_results = {}
    inaccuracy_results = {}
    
    for file_path in comparison_files:
        Cost = MetricsStruct("default", 0)
        Performance = MetricsStruct("default", 0)
        Inaccuracy = MetricsStruct("default", 0)

        CP_Gates, CP_Delay = read_values_from_file(os.path.join(file_path.strip(), "summary/path.txt"))

        try:
            check_string_count(metrics)
        except ValueError as e:
            print(f"Error: {e}")

        if "Gate_Count" in metrics:
            Cost.name = "Gate Count"
            Cost.value = read_csv_value(os.path.join(file_path.strip(), "summary/gates.csv"), 1, 6)
            print(Cost.value)
        elif "Transistor_Count" in metrics:
            Cost.name = "Transistor Count"
            Cost.value = read_csv_value(os.path.join(file_path.strip(), "summary/gates.csv"), 1, 7)
        if "CP_Delay" in metrics:
            Performance.name = "Critical Path Delay"
            Performance.value = CP_Delay
        elif "CP_Gates" in metrics:
            Performance.name = "Critical Path Gate Count"
            Performance.value = CP_Gates
        if "MSE" in metrics:
            Inaccuracy.name = "Mean Square Error"
            Inaccuracy.value = read_csv_value(os.path.join(file_path.strip(), "Error_files/metrics.csv"), 1, 0)
        elif "MAE" in metrics:
            Inaccuracy.name = "Mean Absolute Error"
            Inaccuracy.value = read_csv_value(os.path.join(file_path.strip(), "Error_files/metrics.csv"), 1, 1)
        elif "WCD" in metrics:
            Inaccuracy.name = "Worst-Case Error Distance"
            Inaccuracy.value = read_csv_value(os.path.join(file_path.strip(), "Error_files/metrics.csv"), 1, 2)
        elif "ER" in metrics:
            Inaccuracy.name = "Error Rate"
            Inaccuracy.value = read_csv_value(os.path.join(file_path.strip(), "Error_files/metrics.csv"), 1, 3)
        elif "MHD" in metrics:
            Inaccuracy.name = "Mean Hamming Distance Error"
            Inaccuracy.value = read_csv_value(os.path.join(file_path.strip(), "Error_files/metrics.csv"), 1, 4)
        
        # Store results in respective dictionaries
        cost_results[file_path.strip()] = Cost
        performance_results[file_path.strip()] = Performance
        inaccuracy_results[file_path.strip()] = Inaccuracy


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract all values from dictionaries
    all_costs = [Cost.value for Cost in cost_results.values()]
    all_performances = [Performance.value for Performance in performance_results.values()]
    all_inaccuracies = [Inaccuracy.value for Inaccuracy in inaccuracy_results.values()]

    # Scatter plot all vectors
    ax.quiver(0, 0, 0, all_costs[0], all_performances[0], all_inaccuracies[0], color='red')
    ax.quiver(0, 0, 0, all_costs[1], all_performances[1], all_inaccuracies[1], color='blue')
    ax.quiver(0, 0, 0, all_costs[2], all_performances[2], all_inaccuracies[2], color='purple')
    
    # Orthogonal projections onto axes
    ax.plot([all_costs[0], all_costs[0]], [0, all_performances[0]], [0, 0], color='red', linestyle='--')  
    ax.plot([0, all_costs[0]], [all_performances[0], all_performances[0]], [0, 0], color='red', linestyle='--') 
    ax.plot([all_costs[0], all_costs[0]], [all_performances[0], all_performances[0]], [0, all_inaccuracies[0]], color='red', linestyle='--')  
    
    ax.plot([all_costs[1], all_costs[1]], [0, all_performances[1]], [0, 0], color='blue', linestyle='--')  
    ax.plot([0, all_costs[1]], [all_performances[1], all_performances[1]], [0, 0], color='blue', linestyle='--') 
    ax.plot([all_costs[1], all_costs[1]], [all_performances[1], all_performances[1]], [0, all_inaccuracies[1]], color='blue', linestyle='--')  
    
    ax.plot([all_costs[2], all_costs[2]], [0, all_performances[2]], [0, 0], color='purple', linestyle='--')  
    ax.plot([0, all_costs[2]], [all_performances[2], all_performances[2]], [0, 0], color='purple', linestyle='--') 
    ax.plot([all_costs[2], all_costs[2]], [all_performances[2], all_performances[2]], [0, all_inaccuracies[2]], color='purple', linestyle='--')  
    
    ax.set_xlim(0, 500)  
    ax.set_ylim(0, 40)  
    ax.set_zlim(0, 120)   
    
    # Set labels and title
    ax.set_xlabel(Cost.name)
    ax.set_ylabel(Performance.name)
    ax.set_zlabel(Inaccuracy.name)
    ax.set_title(f"3D Vector Plot ({file_path})")

    ax.legend(comparison_files)
    
    # Show the plot
    plt.show()
    

if __name__ == "__main__":
    main()