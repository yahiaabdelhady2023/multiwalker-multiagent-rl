import os
os.environ["SDL_VIDEODRIVER"] = "dummy"   #allow for headless render so we can only get RBG_ARRAY FOR VIDEO without pop screen
import torch
from pettingzoo.sisl import multiwalker_v9
from tianshou.env import PettingZooEnv, DummyVectorEnv
from tianshou.policy import PPOPolicy, MultiAgentPolicyManager
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
import imageio.v3 as iio
import numpy as np
from tianshou.data import Batch
import numpy as np
from tianshou.data import Batch
import imageio.v3 as iio
import torch
import matplotlib.pyplot as plt

def get_env():
    env = multiwalker_v9.env(n_walkers=3, render_mode="rgb_array")
    return PettingZooEnv(env)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig):
    print("=== Configuration ===")
    print(OmegaConf.to_yaml(cfg))
    print("Parameters used:")
    print("cfg.training.batch_size",cfg.training.batch_size)
    print("cfg.training.max_epoch",cfg.training.max_epoch)
    print("cfg.training.step_per_epoch",cfg.training.step_per_epoch)
    print("cfg.training.step_per_collect",cfg.training.step_per_collect)
    print("cfg.training.repeat_per_collect",cfg.training.repeat_per_collect)
    print("cfg.training.episode_per_test",cfg.training.episode_per_test)


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

    #creating checkpoints!
    hydra_output_dir = HydraConfig.get().runtime.output_dir
    checkpoint_dir = os.path.join(hydra_output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)





    def save_eval_video(video_path, model_path):
        # 1. Switch the shared policy to deterministic evaluation mode
        shared_ppo_policy.eval()

        with iio.imopen(f"{hydra_output_dir}/{video_path}.mp4", "w", plugin="pyav") as vid:
            vid.init_video_stream("libx264", fps=30)

            # 2. Use raw parallel_env with render_mode="rgb_array" for recording
            env = multiwalker_v9.parallel_env(n_walkers=3, render_mode="rgb_array")
            observations, infos = env.reset(seed=42)

            while env.agents:
                # env.agents contains the list of currently active agents (e.g. ['walker_0', 'walker_1', 'walker_2'])
                active_agents = env.agents
                
                # Stack all active observations into shape: (num_active_agents, obs_dim)
                obs_stack = np.array([observations[agent] for agent in active_agents])

                # 3. Create a clean batch without dummy multi-agent keys
                batch = Batch(obs=obs_stack, info=infos)

                # 4. Predict actions for all walkers simultaneously on CUDA
                with torch.no_grad():
                    result = shared_ppo_policy(batch)

                # 5. Move from CUDA -> CPU -> NumPy
                actions_np = result.act.cpu().numpy()

                # 6. Map the actions back to their agent names
                actions = {agent: actions_np[i] for i, agent in enumerate(active_agents)}

                # 7. Step the environment
                observations, rewards, terminations, truncations, infos = env.step(actions)

                # 8. Record frame
                frame = env.render()
                if frame is not None:
                    vid.write_frame(frame)

            env.close()

        # 9. Re-enable training/exploration mode for subsequent epochs
        shared_ppo_policy.train()
        print(f"Video for {model_path} model is saved successfully!")




    def save_reward_team_graph(epoch="final"):
        # 1. Switch the shared policy to deterministic evaluation mode
        shared_ppo_policy.eval()

        # 2. Use parallel_env without rendering for fast evaluation
        env = multiwalker_v9.parallel_env(n_walkers=3)
        observations, infos = env.reset(seed=42)

        cumulative_rewards = []
        step_rewards = []
        current_cumulative = 0.0

        while env.agents:
            active_agents = env.agents
            
            # Stack observations
            obs_stack = np.array([observations[agent] for agent in active_agents])
            batch = Batch(obs=obs_stack, info=infos)

            # Predict actions
            with torch.no_grad():
                result = shared_ppo_policy(batch)

            actions_np = result.act.cpu().numpy()
            actions = {agent: actions_np[i] for i, agent in enumerate(active_agents)}

            # Step environment
            observations, rewards, terminations, truncations, infos = env.step(actions)

            # Calculate total team reward for this step
            team_step_reward = sum(rewards.values())
            current_cumulative += team_step_reward
            
            step_rewards.append(team_step_reward)
            cumulative_rewards.append(current_cumulative)

        env.close()

        # 3. Re-enable training mode
        shared_ppo_policy.train()

        # 4. Plot using matplotlib
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_rewards, label='Cumulative Team Reward', color='blue', linewidth=2)
        plt.plot(step_rewards, label='Per-Step Reward', color='lightblue', alpha=0.7, linewidth=1)
        
        plt.title(f'Multiwalker Team Reward Evaluation (Epoch: {epoch})')
        plt.xlabel('Environment Steps')
        plt.ylabel('Reward')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        # 5. Save graph to the Hydra output directory
        graph_path = os.path.join(hydra_output_dir, f"team_reward_graph_{epoch}.png")
        plt.savefig(graph_path, bbox_inches='tight')
        plt.close()

        print(f"Team reward graph for {epoch} saved successfully to: {graph_path}")
    def save_checkpoint_fn(epoch: int, env_step: int, gradient_step: int):
            save_model_interval = cfg.training.save_model_interval  # Save every X epochs
            
            if epoch > 0 and epoch % save_model_interval == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
                torch.save({
                    'model': marl_policy.state_dict(),
                    'optim': optim.state_dict(), 
                    'epoch': epoch,
                    'env_step': env_step
                }, ckpt_path)

                #saving video as well here
                save_eval_video(f"video_epoch_{epoch}",ckpt_path)
                save_reward_team_graph(epoch)
                return ckpt_path  # <-- THIS MUST BE INDENTED
                
            return None  # <-- Return None when not saving

    log_path = os.path.join(hydra_output_dir, "tb_logs")
    logger = TensorboardLogger(SummaryWriter(log_path))

    # 5. Automated Trainer - SMOKE TEST PARAMETERS
    result = OnpolicyTrainer(
        policy=marl_policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=cfg.training.max_epoch,               # Just run 1 single epoch and exit
        step_per_epoch=cfg.training.step_per_epoch,        # Epoch finishes after only 200 total steps
        step_per_collect=cfg.training.step_per_collect,       # Pause to train every 50 steps
        repeat_per_collect=cfg.training.repeat_per_collect,      # Train on the 50 steps only 1 time
        batch_size=cfg.training.batch_size,             # Tiny batch size
        episode_per_test=cfg.training.episode_per_test,         # Only play 1 test game at the end
        save_checkpoint_fn=save_checkpoint_fn,
        logger=logger
    ).run()

    print(f"Smoke test successful! Best test reward: {result['best_reward']}")

if __name__ == "__main__":
    train()