import torch
from pettingzoo.sisl import multiwalker_v9
from tianshou.env import PettingZooEnv, DummyVectorEnv
from tianshou.policy import PPOPolicy, MultiAgentPolicyManager
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic

def get_env():
    env = multiwalker_v9.env(n_walkers=3)
    return PettingZooEnv(env)

# 1. Environments
train_envs = DummyVectorEnv([get_env for _ in range(8)])
test_envs = DummyVectorEnv([get_env for _ in range(2)])

env_tmp = get_env()
state_shape = env_tmp.observation_space.shape
action_shape = env_tmp.action_space.shape

# 2. Networks (Shrunk to 64x64 for an instant startup)
net_a = Net(state_shape, hidden_sizes=[64, 64], device="cuda")
actor = ActorProb(net_a, action_shape, max_action=env_tmp.action_space.high[0], device="cuda").to("cuda")

net_c = Net(state_shape, hidden_sizes=[64, 64], device="cuda")
critic = Critic(net_c, device="cuda").to("cuda")

optim = torch.optim.Adam(set(actor.parameters()).union(critic.parameters()), lr=3e-4)

# 3. Policy
shared_ppo_policy = PPOPolicy(
    actor=actor,
    critic=critic,
    optim=optim,
    dist_fn=lambda loc, scale: torch.distributions.Normal(loc, scale),
    action_space=env_tmp.action_space,
    action_scaling=True,
    action_bound_method="clip"
)

marl_policy = MultiAgentPolicyManager(
    policies=[shared_ppo_policy, shared_ppo_policy, shared_ppo_policy], 
    env=env_tmp
)

# 4. Collectors (Buffer shrunk for testing)
train_collector = Collector(
    marl_policy, 
    train_envs, 
    VectorReplayBuffer(1000, len(train_envs))
)
test_collector = Collector(marl_policy, test_envs)

# 5. Automated Trainer - SMOKE TEST PARAMETERS
result = OnpolicyTrainer(
    policy=marl_policy,
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=1,               # Just run 1 single epoch and exit
    step_per_epoch=200,        # Epoch finishes after only 200 total steps
    step_per_collect=50,       # Pause to train every 50 steps
    repeat_per_collect=1,      # Train on the 50 steps only 1 time
    batch_size=32,             # Tiny batch size
    episode_per_test=1         # Only play 1 test game at the end
).run()

print(f"Smoke test successful! Best test reward: {result['best_reward']}")