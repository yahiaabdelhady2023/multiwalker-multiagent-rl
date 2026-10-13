# Difference between shared_ppo_policy and marl_policy

1. shared_ppo_policy is like a toolkit , it contains
    - Actor
    - Critic
    - Optimiser (Adam)
    - action_scaling which adds translator that translates output of network [-1,1] to say env specifics like [-5,5] range
    - it has algorithm PPO
    - it has advantage function to calculate advantage value for PPO

2. marl_policy is not an algorithm it is a library technicality to make Tianshou compatiabile with PettingZoo, it has list of shared_ppo_policy
       ` policies=[shared_ppo_policy, shared_ppo_policy, shared_ppo_policy] ` it is not like we are creating 3 separate entities of shared_ppo_policy like 3 separate  actor + critic + algorithm, instead we just use a single network actor + single critic + single optimiser, but the array is like making it compatability with PettingZoo in a sense, we have 3 agents, with agent ids, each agent in env gets its own obs, we need a way to know which agent got this obs and which one needs this output after we feed obs of this specific agent
       to network, so marl_policy acts as bridge to do that for us

3. network itself like actor or Critic does not care about whether we use agent 1 obs or agent 2 obs, we don't feed it agent id we just feed obs to it, it is a mathmatically model that spits out, and is agent-agnositic

# During Evaluation
# Explain how do we eval during Evaluation

1. Pytorch only accepts tensor type not numpy, numpy != tensor
2. Tensor supports CPU, CUDA arrays
3. CPU arrays whether Numpy or Tensor lives in RAM
4. GPU arrays whether  tensor or other dtype lives in VRAM [Video RAM] accessable only by GPU
5. Numpy does not support GPU, only CPU arrays
6. PettingZoo Env has nothing to do all of this
    - PettingZoo observation format is like dict
        {agent_1:[obs_list],agent_2:[obs_list]} 
    - only understands Numpy
7. we want to leverage power of GPU CUDA but at same time make it work with PettingZoo Env How?
8. obs_stack = np.array([observations[agent] for agent in active_agents]) , we stack obs to get (3,obs_dim) array of 3 obs for each agent
9. we need tensor GPU for pytorch, so we pass it to Batch() it converts numpy array to tensor GPU
10. we feed it to pytorch, we get result
11. but we need to feed numpy array action list to pettingzoo and not only that but also we need it in dict format like
    - {'walker_0': actions_np[0], 'walker_1': actions_np[1], 'walker_2': actions_np[2]}
12. We cannot convert from Tensor GPU to numpy CPU, its like french to english but we can convert Tensor CPU to numpy
13. so tensor CUDA GPU --> tensor CPU --> numpy array ,then we format it in dict format as expected by PettingZoo


# AEC Agent Environment Cycle versus Paralleism Envirnoment [Real-time env]

1. AEC is for turn-based, agent 1 obs then takes action, then we go to agent 2 we get obs, and so on, so its turn by turn, you cannot gather obs from all, it is not available, Also AEC is bad for using GPU as you passing single batch for each agent instead of stacking them

2. Paralleism you get access from all info of agents real-time mimicing real time