from sglang.srt.managers.utils import StepMetrics 
import os
import pickle
import sys
from typing import List
import matplotlib.pyplot as plt
import numpy as np
def make_plot(array, title, ylabel, filename):
    
    total_steps = 100 
    
    nparray = np.array(array)
    
    if total_steps < nparray.shape[0]:
        start_idx = (nparray.shape[0] - total_steps) // 2
        nparray = nparray[start_idx:start_idx + total_steps]
    
    maxn = np.max(nparray, axis=1)
    minn = np.min(nparray, axis=1)
    steps = np.arange(nparray.shape[0])

    plt.figure(figsize=(10, 6))
    
    # Plot vertical lines for min-max range at each time step
    plt.vlines(steps, minn, maxn, color='blue', linewidth=2)

    # Add markers for max and min points
    # plt.scatter(steps, maxn, color='red', label='Max', zorder=5, s=1)
    # plt.scatter(steps, minn, color='green', label='Min', zorder=5, s=1)
    
    plt.ylim([np.min(nparray) * 0.9, np.max(nparray) * 1.1])
    
    # Add marks for each item's price at each time step
    # for i in range(nparray.shape[1]):  # Iterate over item IDs (columns)
    #     plt.scatter(steps, nparray[:, i], zorder=4)
        
    # Add labels and title
    plt.xlabel('Steps')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(f"{filename}.png")
    plt.close()
    
def main():
    directory = sys.argv[1]
    metrics_all_ranks: List[List[StepMetrics]] = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'rb') as file:
                metrics_all_ranks.append(pickle.load(file))
                
    nranks = len(metrics_all_ranks)
    
    if nranks == 0:
        print("No data found")
        return
                
    nsteps = len(metrics_all_ranks[0])
    
    if nsteps == 0:
        print("No steps found")
        return
    
    nlayers_per_step = len(metrics_all_ranks[0][0].attention_elapse)
    nexperts_per_rank = len(metrics_all_ranks[0][0].moe_num_tokens_per_local_expert[0])
    nexperts = nexperts_per_rank * nranks
    
    n = nlayers_per_step * nsteps
    
    batch_sizes = [] # n * nranks
    ntokens_per_expert = [] # n * nexperts
    attn_elapse = []
    moe_elapse = []
    
    for i in range(nsteps):
        
        batch_sizes.append([metrics_all_ranks[k][i].batch_size for k in range(nranks)])
        
        for j in range(nlayers_per_step):
            ntokens = []
            attn = []
            moe = []
            for k in range(nranks):
                metric = metrics_all_ranks[k][i]
                ntokens.extend(metric.moe_num_tokens_per_local_expert[j])
                attn.append(metric.attention_elapse[j])
                moe.append(metric.moe_elapse[j])
                
            ntokens_per_expert.append(ntokens)
            attn_elapse.append(attn)
            moe_elapse.append(moe)
            
    make_plot(batch_sizes, 'Batch Size', 'Batch Size', 'batch_size')
    make_plot(ntokens_per_expert, 'Number of Tokens per Expert', 'Number of Tokens', 'ntokens_per_expert')
    make_plot(attn_elapse, 'Attention Elapse Time', 'Time (ms)', 'attn_elapse')
    make_plot(moe_elapse, 'MoE Elapse Time', 'Time (ms)', 'moe_elapse')
                
if __name__ == '__main__':
    main()