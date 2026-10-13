import os
os.environ["SDL_VIDEODRIVER"] = "dummy"   # allow for headless render so we can only get RBG_ARRAY FOR VIDEO without pop screen
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio

from pettingzoo.sisl import multiwalker_v9
from tianshou.env import PettingZooEnv, DummyVectorEnv
from tianshou.policy import PPOPolicy, MultiAgentPolicyManager
from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig



import time

import time

class Stopwatch:
    def __init__(self):
        self._start_time = None
        self._elapsed_time = 0.0

    def start(self):
        """Starts the stopwatch."""
        if self._start_time is not None:
            print("Stopwatch is already running.")
            return

        self._start_time = time.perf_counter()
        print("Stopwatch started...")

    def stop(self):
        """Stops the stopwatch and returns the elapsed time."""
        if self._start_time is None:
            print("Stopwatch is not running.")
            return 0.0

        # Calculate elapsed time
        self._elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None # Reset start time for the next run
        
        # Calculate human-readable components
        hours = int(self._elapsed_time // 3600)
        minutes = int((self._elapsed_time % 3600) // 60)
        seconds = self._elapsed_time % 60
        
        # Format the human-readable string dynamically
        if hours > 0:
            human_readable = f"{hours}h {minutes}m {seconds:.4f}s"
        elif minutes > 0:
            human_readable = f"{minutes}m {seconds:.4f}s"
        else:
            human_readable = f"{seconds:.4f}s"
        
        print(f"Stopwatch stopped. Elapsed time: {self._elapsed_time:.4f} seconds ({human_readable})")
        return self._elapsed_time





def get_env(cfg: DictConfig):
    reward_config = cfg.multiwalker_reward_function
    env = multiwalker_v9.env(
        n_walkers=reward_config.n_walkers,
        forward_reward=reward_config.forward_reward,
        fall_reward=reward_config.fall_reward,
        terminate_reward=reward_config.terminate_reward,
        terminate_on_fall=reward_config.terminate_on_fall,
        shared_reward=reward_config.shared_reward,
        max_cycles=reward_config.max_cycles,
        terrain_length=reward_config.terrain_length,
        render_mode="rgb_array",
    )
    return PettingZooEnv(env)

# --- Feature 1: Custom Logger to intercept and store training rewards ---
class CustomTBLogger(TensorboardLogger):
    def __init__(self, writer, **kwargs):
        super().__init__(writer, **kwargs)
        self.train_rewards_history = []
        self.train_steps_history = []
        
        # Diagnostics histories
        self.update_steps_history = []
        self.kl_divergence_mean_history = []
        self.explained_Variance_mean_history = []
        self.entropy_mean_history = []
        self.clip_loss_mean_history = []
        self.policy_loss_mean_history = []
        self.vf_loss_mean_history = []

    def log_train_data(self, collect_result: dict, step: int) -> None:
        super().log_train_data(collect_result, step)
        if "rew" in collect_result:
            self.train_rewards_history.append(collect_result["rew"])
            self.train_steps_history.append(step)

    def log_update_data(self, update_res: dict, step: int) -> None:
        super().log_update_data(update_res, step)

        kl_temp = []
        ev_temp = []
        ent_temp = []
        clip_temp = []
        loss_temp = []
        vf_temp = []

        for key, value in update_res.items():
            if "explained_variance" in key:
                ev_temp.append(value)
            elif "kl_divergence" in key:
                kl_temp.append(value)
            elif "loss/ent" in key:
                ent_temp.append(value)
            elif "loss/clip" in key:
                clip_temp.append(value)
            elif "loss/vf" in key:
                vf_temp.append(value)
            elif key.endswith("/loss") or key == "loss":
                # Total loss per agent
                loss_temp.append(value)

        # Store step and means only when update metrics are present
        if kl_temp and ev_temp:
            self.update_steps_history.append(step)
            self.explained_Variance_mean_history.append(np.mean(ev_temp))
            self.kl_divergence_mean_history.append(np.mean(kl_temp))
            
            if ent_temp:
                self.entropy_mean_history.append(np.mean(ent_temp))
            if clip_temp:
                self.clip_loss_mean_history.append(np.mean(clip_temp))
            if vf_temp:
                self.vf_loss_mean_history.append(np.mean(vf_temp))
            if loss_temp:
                self.policy_loss_mean_history.append(np.mean(loss_temp))

#custom PPOPolicy
#it is created to extend learn() by adding explained_variance logging and kl_divergent
class CustomPPOPolicy(PPOPolicy):
    def learn(self, batch, batch_size, repeat, **kwargs):
        # 1. Run the standard PPO learn method
        res = super().learn(batch, batch_size, repeat, **kwargs)
        
        # 2. Extract actual returns and Critic predictions from the batch
        returns = batch.returns
        returns_np = returns.detach().cpu().numpy() 
        values = self.critic(batch.obs).detach().cpu().numpy() 

        return_var = np.var(returns_np)
        if return_var == 0:
            explained_var = 0.0
        else:
            explained_var = 1 - (np.var(returns_np - values) / return_var)
            

        res["explained_variance"] = explained_var

        # Calculating KL Divergent Below
        forward_result = self(batch) #feeding it only to PPO network to get probability output
        dist = forward_result.dist
        logp_new = dist.log_prob(batch.act)
        logp_old = torch.tensor(batch.logp_old, device=logp_new.device)     
        approx_kl = (logp_old - logp_new).mean().item()
        res["kl_divergence"] = approx_kl
        return res


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig):
    print("=== Configuration ===")
    print(OmegaConf.to_yaml(cfg))
    print("Parameters used:")
    print("cfg.training.batch_size", cfg.training.batch_size)
    print("cfg.training.max_epoch", cfg.training.max_epoch)
    print("cfg.training.step_per_epoch", cfg.training.step_per_epoch)
    print("cfg.training.step_per_collect", cfg.training.step_per_collect)
    print("cfg.training.repeat_per_collect", cfg.training.repeat_per_collect)
    print("cfg.training.episode_per_test", cfg.training.episode_per_test)

    # 1. Environments
    train_envs = DummyVectorEnv([lambda: get_env(cfg) for _ in range(8)])
    test_envs = DummyVectorEnv([lambda: get_env(cfg) for _ in range(2)])

    env_tmp = get_env(cfg)
    state_shape = env_tmp.observation_space.shape
    action_shape = env_tmp.action_space.shape

    # 2. Networks
    net_a = Net(state_shape, hidden_sizes=[64, 64], device="cuda")
    actor = ActorProb(net_a, action_shape, max_action=env_tmp.action_space.high[0], device="cuda").to("cuda")

    net_c = Net(state_shape, hidden_sizes=[64, 64], device="cuda")
    critic = Critic(net_c, device="cuda").to("cuda")

    optim = torch.optim.Adam(set(actor.parameters()).union(critic.parameters()), lr=3e-4)

    # 3. Policy
    shared_ppo_policy = CustomPPOPolicy(
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

    # 4. Collectors
    train_collector = Collector(
        marl_policy, 
        train_envs, 
        VectorReplayBuffer(1000, len(train_envs))
    )
    test_collector = Collector(marl_policy, test_envs)

    # Setup Checkpoint Directory
    hydra_output_dir = HydraConfig.get().runtime.output_dir
    checkpoint_dir = os.path.join(hydra_output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Setup Custom Logger
    log_path = os.path.join(hydra_output_dir, "tb_logs")
    writer = SummaryWriter(log_path)
    logger = CustomTBLogger(writer,
                               train_interval=10,
                               update_interval=10,
                               test_interval=10)

    def save_eval_video(epoch_dir):
        shared_ppo_policy.eval()
        video_path = os.path.join(epoch_dir, "eval_video.mp4")

        with iio.imopen(video_path, "w", plugin="pyav") as vid:
            vid.init_video_stream("libx264", fps=30)
            env = multiwalker_v9.parallel_env(
                n_walkers=cfg.multiwalker_reward_function.n_walkers,
                forward_reward=cfg.multiwalker_reward_function.forward_reward,
                fall_reward=cfg.multiwalker_reward_function.fall_reward,
                terminate_reward=cfg.multiwalker_reward_function.terminate_reward,
                terminate_on_fall=cfg.multiwalker_reward_function.terminate_on_fall,
                shared_reward=cfg.multiwalker_reward_function.shared_reward,
                max_cycles=cfg.multiwalker_reward_function.max_cycles,
                terrain_length=cfg.multiwalker_reward_function.terrain_length,
                render_mode="rgb_array",
            )
            observations, infos = env.reset(seed=42)

            while env.agents:
                active_agents = env.agents
                obs_stack = np.array([observations[agent] for agent in active_agents])
                batch = Batch(obs=obs_stack, info=infos)

                with torch.no_grad():
                    result = shared_ppo_policy(batch)

                actions_np = result.act.cpu().numpy()
                actions = {agent: actions_np[i] for i, agent in enumerate(active_agents)}

                observations, rewards, terminations, truncations, infos = env.step(actions)

                frame = env.render()
                if frame is not None:
                    vid.write_frame(frame)

            env.close()

        shared_ppo_policy.train()
        print(f"Video saved successfully to: {video_path}")

    def save_training_graph(epoch, epoch_dir):
            # --- Plot 1: Mean Training Team Reward ---
            plt.figure(figsize=(10, 6))
            if logger.train_steps_history and logger.train_rewards_history:
                plt.plot(logger.train_steps_history, logger.train_rewards_history, 
                        label='Mean Batch Reward (Training)', color='orange', linewidth=2)
                
            plt.title(f'Multiwalker Training Reward (Up to Epoch: {epoch})')
            plt.xlabel('Environment Steps')
            plt.ylabel('Reward')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()

            graph_path = os.path.join(epoch_dir, "training_team_reward_graph.png")
            plt.savefig(graph_path, bbox_inches='tight')
            plt.close()
            print(f"Training reward graph saved successfully to: {graph_path}")

            # --- Plot 2: 3x2 Policy Diagnostics Grid ---
            if getattr(logger, "update_steps_history", None) and logger.update_steps_history:
                steps = logger.update_steps_history
                fig, axes = plt.subplots(3, 2, figsize=(15, 12))

                # 1. Explained Variance
                axes[0, 0].plot(steps, logger.explained_Variance_mean_history, color='teal', linewidth=1.5)
                axes[0, 0].axhline(0, color='gray', linestyle=':', alpha=0.7)
                axes[0, 0].axhline(1, color='green', linestyle=':', alpha=0.7, label='Optimal (1.0)')
                axes[0, 0].set_title('Mean Explained Variance')
                axes[0, 0].set_ylabel('Explained Var')
                axes[0, 0].grid(True, linestyle='--', alpha=0.5)
                axes[0, 0].legend(loc='lower right')

                # 2. Approximate KL Divergence
                axes[0, 1].plot(steps, logger.kl_divergence_mean_history, color='crimson', linewidth=1.5)
                axes[0, 1].axhline(0, color='gray', linestyle=':', alpha=0.7)
                axes[0, 1].set_title('Mean Approx KL Divergence')
                axes[0, 1].set_ylabel('KL Divergence')
                axes[0, 1].grid(True, linestyle='--', alpha=0.5)

                # 3. Policy Entropy
                if logger.entropy_mean_history:
                    axes[1, 0].plot(steps, logger.entropy_mean_history, color='darkviolet', linewidth=1.5)
                    axes[1, 0].set_title('Mean Policy Entropy')
                    axes[1, 0].set_ylabel('Entropy')
                    axes[1, 0].grid(True, linestyle='--', alpha=0.5)

                # 4. Clipping Loss (Surrogate Loss)
                if logger.clip_loss_mean_history:
                    axes[1, 1].plot(steps, logger.clip_loss_mean_history, color='royalblue', linewidth=1.5)
                    axes[1, 1].set_title('Mean Policy Loss (Clip)')
                    axes[1, 1].set_ylabel('Clip Loss')
                    axes[1, 1].grid(True, linestyle='--', alpha=0.5)

                # 5. Value Function (Critic) Loss
                if logger.vf_loss_mean_history:
                    axes[2, 0].plot(steps, logger.vf_loss_mean_history, color='darkorange', linewidth=1.5)
                    axes[2, 0].set_title('Mean Value Function Loss (Critic)')
                    axes[2, 0].set_xlabel('Update Steps')
                    axes[2, 0].set_ylabel('VF Loss')
                    axes[2, 0].grid(True, linestyle='--', alpha=0.5)

                # 6. Overall Combined Loss
                if logger.policy_loss_mean_history:
                    axes[2, 1].plot(steps, logger.policy_loss_mean_history, color='brown', linewidth=1.5)
                    axes[2, 1].set_title('Mean Total Loss')
                    axes[2, 1].set_xlabel('Update Steps')
                    axes[2, 1].set_ylabel('Total Loss')
                    axes[2, 1].grid(True, linestyle='--', alpha=0.5)

                plt.suptitle(f'PPO Training Diagnostics (Epoch: {epoch})', fontsize=16, y=0.995)
                plt.tight_layout()
                
                diag_path = os.path.join(epoch_dir, "policy_diagnostics_grid.png")
                plt.savefig(diag_path, bbox_inches='tight')
                plt.close()
                print(f"Diagnostics grid saved successfully to: {diag_path}")

    def save_reward_team_graph_eval(epoch, epoch_dir):
        shared_ppo_policy.eval()
        env = multiwalker_v9.parallel_env(
            n_walkers=cfg.multiwalker_reward_function.n_walkers,
            forward_reward=cfg.multiwalker_reward_function.forward_reward,
            fall_reward=cfg.multiwalker_reward_function.fall_reward,
            terminate_reward=cfg.multiwalker_reward_function.terminate_reward,
            terminate_on_fall=cfg.multiwalker_reward_function.terminate_on_fall,
            shared_reward=cfg.multiwalker_reward_function.shared_reward,
            max_cycles=cfg.multiwalker_reward_function.max_cycles,
            terrain_length=cfg.multiwalker_reward_function.terrain_length,
        )
        observations, infos = env.reset(seed=42)

        cumulative_rewards = []
        step_rewards = []
        current_cumulative = 0.0

        while env.agents:
            active_agents = env.agents
            obs_stack = np.array([observations[agent] for agent in active_agents])
            batch = Batch(obs=obs_stack, info=infos)

            with torch.no_grad():
                result = shared_ppo_policy(batch)

            actions_np = result.act.cpu().numpy()
            actions = {agent: actions_np[i] for i, agent in enumerate(active_agents)}

            observations, rewards, terminations, truncations, infos = env.step(actions)

            team_step_reward = sum(rewards.values())
            current_cumulative += team_step_reward
            
            step_rewards.append(team_step_reward)
            cumulative_rewards.append(current_cumulative)

        env.close()
        shared_ppo_policy.train()

        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_rewards, label='Cumulative Team Reward', color='blue', linewidth=2)
        plt.plot(step_rewards, label='Per-Step Reward', color='lightblue', alpha=0.7, linewidth=1)
        
        plt.title(f'Multiwalker Team Eval Reward (Epoch: {epoch})')
        plt.xlabel('Environment Steps')
        plt.ylabel('Reward')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        graph_path = os.path.join(epoch_dir, "eval_team_reward_graph.png")
        plt.savefig(graph_path, bbox_inches='tight')
        plt.close()

        print(f"Eval team reward graph saved successfully to: {graph_path}")

    # --- Feature 2: Organized Folder Structure ---
    def save_checkpoint_fn(epoch: int, env_step: int, gradient_step: int):
        # Fallback to 1 if not set in config, to avoid KeyErrors during smoke tests
        save_model_interval = getattr(cfg.training, "save_model_interval", 1) 
        
        if epoch > 0 and epoch % save_model_interval == 0:
            # Create isolated folder for this specific epoch
            epoch_dir = os.path.join(checkpoint_dir, f"epoch_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
            
            # Save model weights
            ckpt_path = os.path.join(epoch_dir, "model.pth")
            torch.save({
                'model': marl_policy.state_dict(),
                'optim': optim.state_dict(), 
                'epoch': epoch,
                'env_step': env_step
            }, ckpt_path)

            # Generate and save metrics into the epoch folder
            save_eval_video(epoch_dir)
            save_reward_team_graph_eval(epoch, epoch_dir)
            save_training_graph(epoch, epoch_dir)
            
            return ckpt_path
            
        return None 

    # 5. Automated Trainer
    result = OnpolicyTrainer(
        policy=marl_policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=cfg.training.max_epoch,
        step_per_epoch=cfg.training.step_per_epoch,
        step_per_collect=cfg.training.step_per_collect,
        repeat_per_collect=cfg.training.repeat_per_collect,
        batch_size=cfg.training.batch_size,
        episode_per_test=cfg.training.episode_per_test,
        save_checkpoint_fn=save_checkpoint_fn,
        logger=logger
    ).run()

    print(f"Best test reward: {result['best_reward']}")

if __name__ == "__main__":

    timer = Stopwatch()
    timer.start()
    train()
    timer.stop()