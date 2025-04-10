from sglang.srt.managers.utils import StepMetrics 
import os
import pickle
import sys
from typing import List
import matplotlib.pyplot as plt
import numpy as np

def heatmap(data, title, label, save_path):
    data = np.transpose(data)
    # use pyplot to draw a heatmap of global_expert_num_tokens
    plt.figure(figsize=(10, 6))
    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.title(title)
    plt.xlabel("Layer ID")
    plt.ylabel("Expert ID")
    plt.colorbar(label=label)
    plt.xticks(np.arange(data.shape[1]), np.arange(data.shape[1]))
    plt.yticks(np.arange(data.shape[0]), np.arange(data.shape[0]))
    plt.grid(False)
    plt.savefig(save_path, dpi=300)

def main():
    directory = sys.argv[1]
    metrics_all_ranks: List[List[StepMetrics]] = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if not filename.endswith(".pickle"):
            continue
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
    
    print("nranks: ", nranks)
    print("nsteps: ", nsteps)
    print("nlayers: ", nlayers_per_step)
    print("nexperts: ", nexperts)
    
    n = nlayers_per_step * nsteps
    
    batch_sizes = [] # n * nranks
    is_decode = [] # n * nranks
    ntokens_per_expert = [] # n * nexperts
    attn_elapse = []
    moe_elapse = []
    attn_all_gather_elapse = []
    all_gather_elapse = []
    
    decode_steps = []
    
    for i in range(nsteps):
        
        batch_sizes.append([metrics_all_ranks[k][i].batch_size for k in range(nranks)])
        is_decode.append([metrics_all_ranks[k][i].is_decode for k in range(nranks)])
        
        step_is_decode = all(is_decode)
        if step_is_decode:
            decode_steps.append(i)
        
        for j in range(nlayers_per_step):
            ntokens = []
            attn = []
            moe = []
            all_gather = []
            attn_all_gather = []
            for k in range(nranks):
                metric = metrics_all_ranks[k][i]
                ntokens.extend(metric.moe_num_tokens_per_local_expert[j])
                attn.append(metric.attention_elapse[j])
                moe.append(metric.moe_elapse[j])
                attn_all_gather.append(metric.attention_elapse[j] + metric.all_gather_elapse[j])
                all_gather.append(metric.all_gather_elapse[j])
                
            ntokens_per_expert.append(ntokens)
            attn_elapse.append(attn)
            moe_elapse.append(moe)
            all_gather_elapse.append(all_gather)
            attn_all_gather_elapse.append(attn_all_gather)
            
    print(f"decode steps: {decode_steps}")
            
    expert_num_tokens = np.array(ntokens_per_expert).reshape((nsteps, nlayers_per_step, -1))
    moe_elapse = np.array(moe_elapse).reshape((nsteps, nlayers_per_step, -1))
    global_expert_num_tokens = expert_num_tokens.sum(axis=0) / 1000
    
    decode_expert_num_tokens = expert_num_tokens[decode_steps]
    decode_moe_elapse = moe_elapse[decode_steps]
    decode_global_expert_num_tokens = decode_expert_num_tokens.sum(axis=0) / 1000
    
    def get_path(filename):
        return os.path.join(directory, filename)
    
    def get_decode_path(filename):
        return os.path.join(directory, f"decode_{filename}")
    
    heatmap(global_expert_num_tokens, "global #tokens per expert", "#thousand tokens", get_path("global_expert_num_tokens.png"))
    heatmap(expert_num_tokens[100], "one batch #tokens per expert", "#tokens", get_path("one_batch_expert_num_tokens.png"))
    heatmap(moe_elapse[100], "one batch execution time per expert", "time cost(ms)", get_path("moe_elapse.png"))
    
    heatmap(decode_global_expert_num_tokens, "decode #tokens per expert", "#thousand tokens", get_decode_path("decode_global_expert_num_tokens.png"))
    heatmap(decode_expert_num_tokens[50], "decode one batch #tokens per expert", "#tokens", get_decode_path("decode_one_batch_expert_num_tokens.png"))
    heatmap(decode_moe_elapse[50], "decode one batch execution time per expert", "time cost(ms)", get_decode_path("decode_moe_elapse.png"))
    
if __name__ == '__main__':
    main()