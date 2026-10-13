## How does PPO works?

## Summary of how does PPO works

there are `Actor Network` and `Critic Network` on a basic level, Before Actor does anything, Critic predicts next state baseline average reward then Actor predicts probabilty for each action, selects one
receives reward, Advantage function is used to calculate the difference between baseline predicted reward vs true reward
both networks uses it to calculate their losses and then update network using clip mechanism to prevent catescrophic forgeting issue

## In details how does PPO works

### 1. Data Collecting Phase

Critic predicts next state baseline average reward, Actor predicts probabilty for each action select ones
, actor observes new reward and new state, this is a single experience that will be stored in rollout buffer
until it is full or after total number of time steps has passed

### 2. Advantage Value Calculating Phase

Advantage function is used to evaluate each experience [difference between predicted reward versus true reward value]
and it spits out advantage value that is either positive or negative advantage value, what's the difference?

#### Advantage value tells us: Is our baseline reward estimate, an overestimate or underestimate?

#### Positive Advantage value: it is basically saying we underestimated our baseline value, we should increase our future baseline reward value for this state and increase likehood of selecting this action (increase probability)

#### negative Advantage value: it is basically saying we overestimated our baseline value, we should decrease our future baseline reward value for this state and decrease likehood of selecting this action (decrease probability)


### 3. Mini Batches Phase

if our rollout buffer size is 10k, and batch_size is 256, we have 39 mini batches, we use all mini batches during training, now we dont just train once using this rollout buffer but rather over several epoches using same data, but isn't that bad what if our agent memories this data or overfits? that used to be case with old RL algorithms but THIS TAKES us
to next phase that is `Clipping mechanism Phase`

### 4. Loss function and Clipping Mechanism phase

we use a clipped Actor loss [policy loss] because Actor spits out probability so it is a policy network, and clipped Critic Loss [Value loss] because Critic spits out reward predictions similar to DQN or Q-Learning Q values, why are both clipped, clipped here means

#### Clipped: in simple terms we check new update and old update check difference, if its too big we do partial update, like if it new policy is difference than old one by 25% , and our threshold is 20% we do for 20% update a parially one

### 5. Backprogation

we use loss and feed it to Adam optimiser for example, to calculate gradients we use then to adjust weights accordingly

### 6. On-Policy Explained

it is on-policy why? because rollout buffer is cleared after we finish training using it, in DQN we don't clear it
we keep it so we can learn from previous experiences very ancient one hence off-policy term, on stands for learning immediate, philosphy behind PPO is that i don't need to look at my experiences say 100 epochs ago if i was having dumber
brain or younger, i should focus on learning from my present mistakes

### 7. Entropy term

Entropy term is a fancy word for randomness or exploration likeihood, it is integreted specifically into Actor clipped policy function, initially Entropy is high that is to say we give good reward in loss function to encourge it to have agent actor spit out approximately evenly probabily distribution for each action e.g. 33%/33%/33% is high Entropy, but over time we decrease it until agent converges to specific action like 90%/5%/5% that is Low Entropy


### KL (Kullback-Leibler) // KL Divergence

that was old school mathimatically way to calculate difference between two probability distribution, that is to say
that is similar or exact same mechanism as our Clipped mechanism, however it was slow, clipped mechanism is faster
way of doing same thing, there is PPO-Penalty is version of PPO that still uses KL divergence instead of modern clipped mechanism, OpenAI recommends PPO-Clip that is industry standard!


### Important: Why do we train multiple times using same rollout buffer?

in our previous example we had 10k rollout buffer size, and 256 batch size, and 39 mini batches, 10k/256 = 39 mini batches, we don't just train once or iterative once over all these 39 mini batches, we repeat that several epoches, why?

1. to make best use of experience, we have safety built feature that prevents agent from overmemorising these experiences or forgeting its current strategy, instead each time we train we slightly change its brain, so it is efficient to do that
