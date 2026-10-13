# Experiment One

## Hyperparameters: 
training:
  batch_size: 512 
  max_epoch: 5
  step_per_epoch: 200000    
  step_per_collect: 2048      
  repeat_per_collect: 3    
  episode_per_test: 1
  save_model_interval: 1   
  learning_rate: 0.0003 
multiwalker_reward_function:
  n_walkers: 3 #default value is 3
  forward_reward: 1.0 #default value is 1
  fall_reward: -10.0 #default value is -10
  terminate_reward: -100.0 #default value is -100.0
  terminate_on_fall: True #default value is True
  shared_reward: True #default value is True
  max_cycles: 500 #default value is 500
  terrain_length: 200 #default value is 200


## Hyperparameters Overview:
These will severe as default baseline values, this is the first experiment, it took about 11 minutes to train,
Total number of time steps = step_per_epoch * max_epoch = 1000000 ( 1 Million), each experiment will be trained for 1 Million total time steps, learning rate is 0.0003 is known as magically number recommended by OpenAI for Adam in RL, it is used here, Batch_size is 512, it should be big, it is better for GPU, modern ones to take big numbers than lots of small batchs, that is going to slow down GPU and training

#### Note Rollout Buffer is not same as reply buffer size used by DQN, the difference is that, Rollout buffer is used PPO and it in our case its length is 16384 , where did that come from?, we start training every 2048 time step, after each 2048 time steps pass we start training, but since we have 8 env, we collect 8x2048 = 16384, then once we use these slots to update network we empty it all, philosphy behind PPO rollout buffer is that i dont need old experience when i was making silly mistakes early on, unlike DQN replay buffer that has usually big buffer size say of 1 million, and it is off-policy learning meaning we dont need to immediate data we collected to train, we could use experience from very ancient episode to train and it only removes old data to replace part of it with new one , once its full, while PPO is on-poilcy it learns on spot then empties out buffer

regarding multiwalker env variable values these are default one from their page



## Video Analysis and Graph Analysis
we have 5 models, 5 videos and 5 graphs per each model

### Epoch 1 Model

<video width="600" controls>
  <source src="../outputs/2026-10-11/23-12-47/checkpoints/epoch_1/eval_video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

#### 1. Agents Behaviors
agent seems to struggle to walk early on, all are falling at the same time

#### 2. Graph

![Epoch 1 evaluation team reward graph](../outputs/2026-10-11/23-12-47/checkpoints/epoch_1/training_team_reward_graph.png)

initally agent actually gets good results but it seems to struggle later
one, and get worst results after, but it smooths out

### Epoch 2 Model

<video width="600" controls>
  <source src="../outputs/2026-10-11/23-12-47/checkpoints/epoch_2/eval_video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

#### 1. Agents Behaviors
agents learnt seemed to learn that falling or triping over is bad, even tho mid agent died, they seem to trying to avoid falling, altho they don't move forward

#### 2. Graph

![Epoch 2 team reward graph](../outputs/2026-10-11/23-12-47/checkpoints/epoch_2/training_team_reward_graph.png)

starting 3 Million to around 4 Million time steps,
agent was getting better rewards, and less penalities

### Epoch 3 Model

<video width="600" controls>
  <source src="../outputs/2026-10-11/23-12-47/checkpoints/epoch_3/eval_video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

#### 1. Agents Behaviors
Agents seemed to forget how to balance although compared to epoch 1 they are trying to move forward, and but they seemed
to forget how to avoid falling

#### 2. Graph

![Epoch 3 team reward graph](../outputs/2026-10-11/23-12-47/checkpoints/epoch_3/training_team_reward_graph.png)

starting from end of previous epoch, we see agent reward values became worse even worse than epoch 1,
agent didn't keep in its memory how to balance learnt in previous epoch, it is trying to moving forward

### Epoch 4 Model

<video width="600" controls>
  <source src="../outputs/2026-10-11/23-12-47/checkpoints/epoch_4/eval_video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

#### 1. Agents Behaviors
it is hard to tell what is the agent trying to do, but it could that agents are trying to move forward while balancing altho performance is worse than in epoch 2

#### 2. Graph

![Epoch 4 team reward graph](../outputs/2026-10-11/23-12-47/checkpoints/epoch_4/training_team_reward_graph.png)

still agent gets around -120 reward similar to previous epoch 3, altho with slight improvement in graph, and agents behavior in video, not just moving forrward without any attempt to balance

### Epoch 5 Model

<video width="600" controls>
  <source src="../outputs/2026-10-11/23-12-47/checkpoints/epoch_5/eval_video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

#### 1. Agents Behaviors
it looks like the agent is trying to priorities 
staying alive than moving forward, by just crouching
 altho mid agent just like in epoch 2 died, but 
 it is going back to old strategy

#### 2. Graph

![Epoch 5 team reward graph](../outputs/2026-10-11/23-12-47/checkpoints/epoch_5/training_team_reward_graph.png)

looks very identical that part of graph to epoch 2,
getting small peaks and better reward values than most episodes


## Conclusion:

agent seems to be trying out initally balancing and not moving but in between it switched to aggressive moving forward strategy without learning how to balance + walk, then it pivot back to balancing strategy because it yield much better results than dying from moving forward


## Parameters to possibly tweak

1. we can play with fall_reward and increase it from -10 to -15
   - justification: agent keeps falling a lot, we could encourge it to stay alive by giving bigger penality