# Experiment Two

## Hyperparameters

```yaml
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
  n_walkers: 3
  forward_reward: 1.0
  fall_reward: -15.0
  terminate_reward: -100.0
  terminate_on_fall: true
  shared_reward: true
  max_cycles: 500
  terrain_length: 200
```

## Hyperparameters Overview

same as experiment one however the fall_reward has increased from -10 to -15

## Video Analysis and Graph Analysis

### Epoch 1 Model

<video width="600" controls>
  <source src="../outputs/2026-10-12/00-34-55/checkpoints/epoch_1/eval_video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

#### 1. Agents Behaviors
agent didn't learn how to balance properly, all agents felt straight down

#### 2. Graph

![Epoch 1 team reward graph](../outputs/2026-10-12/00-34-55/checkpoints/epoch_1/training_team_reward_graph.png)

early on agent is still playing and the graph seems unstable

### Epoch 2 Model

<video width="600" controls>
  <source src="../outputs/2026-10-12/00-34-55/checkpoints/epoch_2/eval_video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

#### 1. Agents Behaviors
agent behavior is still the same,

#### 2. Graph

![Epoch 2 team reward graph](../outputs/2026-10-12/00-34-55/checkpoints/epoch_2/training_team_reward_graph.png)

rewards that were collected during this epoch are worse than first epoch

### Epoch 3 Model

<video width="600" controls>
  <source src="../outputs/2026-10-12/00-34-55/checkpoints/epoch_3/eval_video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

#### 1. Agents Behaviors
agent behavior is still the same

#### 2. Graph

![Epoch 3 team reward graph](../outputs/2026-10-12/00-34-55/checkpoints/epoch_3/training_team_reward_graph.png)

rewards that were collected during this epoch are worse than first epoch


### Epoch 4 Model

<video width="600" controls>
  <source src="../outputs/2026-10-12/00-34-55/checkpoints/epoch_4/eval_video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

#### 1. Agents Behaviors

agent is learning currently how to balance altho it seems to fail, but it does that by
crouching not standing or moving forward

#### 2. Graph

![Epoch 4 team reward graph](../outputs/2026-10-12/00-34-55/checkpoints/epoch_4/training_team_reward_graph.png)

reward seems better than before a lot, in fact this probably is one of the best epochs

### Epoch 5 Model

<video width="600" controls>
  <source src="../outputs/2026-10-12/00-34-55/checkpoints/epoch_5/eval_video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

#### 1. Agents Behaviors

agent behavior is best here overall, the episode length is longer in video its 2 seconds,
it is actively trying to balance, altho agent in mid always seems to die first

#### 2. Graph

![Epoch 5 team reward graph](../outputs/2026-10-12/00-34-55/checkpoints/epoch_5/training_team_reward_graph.png)

graph for this specific period seems much better than previous ones, in fact it has probably one of best rewards across all epochs


## Conclusion:

agent initially was trying to learn
how to balance but it failed several times until in Epoch 4, it started to actively trying to do it, and so it did better balancing in Epoch 5, altho it keeps dying especially the mid agent, increasing penality seemed to help, but we want to encourge to avoid dying early one, I suggest increasing terminate reward to something big like -150 and fall_reward say to -25 from -15



## Parameters to possibly tweak

1. we can play with fall_reward and increase it from -15 to -25
   - justification: to ensure the agent priorities balancing we need a bigger penality for falling
  
2. increase terminate_reward from -100 to -150
    - justification: to ensure the agent priorities balancing we need a bigger penality for falling
