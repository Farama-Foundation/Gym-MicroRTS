import numpy as np
import torch
from torch.distributions import Categorical
from matplotlib import pyplot as plt

initial_agent_quality = 1
past_agent_pools = [['agent1', initial_agent_quality]]
eta = 0.02

fig, axes = plt.subplots(ncols=2, nrows=2)
axe_idx = 0
winrate = 1
for version in range(1, 2500):
    quality_scores = torch.tensor(np.array(past_agent_pools)[:,1].astype(np.float32))
    quality_probs = Categorical(quality_scores)
    sampled_agent_idx = quality_probs.sample().item()
    if np.random.rand() > (1-winrate):
        past_agent_pools[sampled_agent_idx][1] = max(1e-3, quality_scores[sampled_agent_idx].item() - eta / (len(past_agent_pools)* quality_probs.probs[sampled_agent_idx].item()))
    if version % 10 == 0:
        past_agent_pools += [[f'agent{version}', quality_scores.max().item()]]
        
    if version % 500 == 0:
        axes.flatten()[axe_idx].plot(Categorical(quality_scores[-50:]).probs)
        axes.flatten()[axe_idx].set_ylim([0, 0.05])
        axe_idx += 1
        # winrate = max(0.5, winrate - 0.15)

    

# axes.flatten()[axe_idx].plot(Categorical(quality_scores).probs)