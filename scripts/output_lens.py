import json
import matplotlib.pyplot as plt
import sys

def create_histogram(json_file_path):
    # Load JSON data
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Extract output_lens values
    output_lens = data['output_lens']
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(output_lens, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Output Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Output Lengths')
    plt.grid(True, alpha=0.3)
    plt.savefig('output_lens_dist.png', dpi=300, bbox_inches='tight')
    plt.show()

# Usage
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python output_lens_dist.py <json_file_path>")
        sys.exit(1)
    
    json_file_path = sys.argv[1]
    create_histogram(json_file_path)