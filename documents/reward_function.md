## Reward Function Parameters:

1. forward_reward (DEFAULT = 1)
    - it is forward_reward * (change in package X position)
    - in other words if forward_reward=1, and package was at x=0 and now is at x=50, we get 50 REWARD
    - if forward_reward =1 and package at x = 0 and package moved 0.1, you get 0.1 reward
    - it is relative to movement of box rather than absolute value

2. fall_reward (DEFAULT = -10)
    - agent receives -10 if they fall whether terminate_reward is True or False

3. terminate_reward (DEFAULT = -100)
    - only works if terminate_on_fall is True,
    - if agent falls all agents receives -100 given that shared_reward = True, and episode terminates

4. max_cycles (DEFAULT = 500)
    - when should episode terminate?, if 500 time steps passes


## How does agent get shared_reward?

1. 3 agents, 1 falls gets -10 fall reward, but terminate_on_fall=True, so also gets -100, so
that agent got -110, but the other two agents get -100 because terminate_on_fall = True, so
shared_reward = -110 + -100 + -100 = -310 for all

## notes for changing variables


1. for forward_reward as we will frequently get it, it should be small, why?
    - if our network has to predict [critic] and its prediction range is small, error will be small with target, small error small weight update, but if it is big we get big error hence big update aka making agent forget what it learnt
    - so forward_reward should remain [0.25,0.5,etc... max until 10]

2. Terminate Reward
    - it is expectation because we get it barely not often so it should be big, altho it can be -50 or -25,

3. everything overall should be between -100 to + 10, to keep relatively working and math make sense


4. use 2x,3x,4x but stay within limit and make sure forward_reward doesn't exceed 10