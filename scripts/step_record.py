import json
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python output_lens_dist.py <json_file_path>")
        sys.exit(1)

    json_file_path = sys.argv[1]

    # Load JSON file
    with open(json_file_path, "rt") as f:
        data = json.load(f)

    # Extract data
    timestamps = [entry["timestamp"] for entry in data]
    batch_sizes = [entry["batch_size"] for entry in data]
    durations = [entry["duration"] for entry in data]

    # Check is timestamps are ascending order
    if not all(timestamps[i] <= timestamps[i + 1] for i in range(len(timestamps) - 1)):
        raise ValueError("Timestamps are not in ascending order.")

    # Create plot with dual y-axes
    fig, ax1 = plt.subplots()

    # Left Y-axis: Batch Size
    color1 = 'tab:blue'
    ax1.set_xlabel('Timestamp (s)')
    ax1.set_ylabel('Batch Size', color=color1)
    ax1.plot(timestamps, batch_sizes, color=color1, label='Batch Size')
    ax1.tick_params(axis='y', labelcolor=color1)

    # Right Y-axis: Duration (dotted line)
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Duration (s)', color=color2)
    ax2.plot(timestamps, durations, linestyle=':', color=color2, label='Duration')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Title and layout
    plt.title('Batch Size and Duration Over Time')
    fig.tight_layout()
    plt.savefig("batch_size_duration_plot.png", dpi=300, bbox_inches='tight')