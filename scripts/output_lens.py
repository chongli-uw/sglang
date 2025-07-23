import json
import matplotlib.pyplot as plt
import sys
import numpy as np
from collections import defaultdict

def create_histogram(json_file_path):
    # Load JSON data
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # get the dir path of json file, consider that the path is just the file under current directory
    # and not a nested directory structure
    dir_path = json_file_path.rsplit('/', 1)[0]
    if dir_path == json_file_path:
        dir_path = '.'

    # Extract output_lens values
    output_lens = data['output_lens']

    # bin count output_lens
    max_output_len = max(output_lens)
    num_seqs_by_output_lens = [0] * (max_output_len + 1)
    for lens in output_lens:
        num_seqs_by_output_lens[lens] += 1
    
    num_remaining_seqs = [0] * (max_output_len + 1)
    num_remaining_seqs[0] = len(output_lens)  # All sequences are remaining at output length 0
    for i in range(1, max_output_len + 1):
        num_remaining_seqs[i] = num_remaining_seqs[i - 1] - num_seqs_by_output_lens[i - 1]
    
    # Create histogram
    # plt.figure(figsize=(10, 6))
    fig, ax1 = plt.subplots()

    ax1.hist(output_lens, bins=30, color="#FFDDC1", edgecolor="#8B0000", alpha=0.7)
    ax1.set_xlabel('Output Length')
    ax1.set_ylabel('Frequency')
    ax1.set_xlim(left=0)

    # Sample 30 points from num_remaining_seqs and plot them
    remaininig_seqs_color = "#003366"
    ax2 = ax1.twinx()
    ax2.set_ylabel('Remaining Sequences', color=remaininig_seqs_color)
    sample_indices = list(range(0, max_output_len + 1, max_output_len // 30))  # Sample every nth index
    sampled_remaining = [num_remaining_seqs[i] for i in sample_indices]
    indices = np.array(sample_indices)
    # make ax2 plot log-scale
    ax2.set_yscale('log')
    ax2.plot(indices + (max_output_len // 60), sampled_remaining, color=remaininig_seqs_color, marker='.', linestyle='-', label='Remaining Sequences', linewidth=1, markersize=2)
    ax2.tick_params(axis='y', labelcolor=remaininig_seqs_color)
    ax2.set_ylim(bottom=0)

    plt.title('Distribution and Remaining Sequences by Output Lengths')
    # plt.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(f'{dir_path}/output_lens_dist.png', dpi=300, bbox_inches='tight')
    plt.show()

# Usage
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python output_lens_dist.py <json_file_path>")
        sys.exit(1)
    
    json_file_path = sys.argv[1]
    create_histogram(json_file_path)