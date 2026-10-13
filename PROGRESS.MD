

# 10/5/2026

1. Organised src file, separated training from src
2. searched and downloaded tianshou library instead of rayLIB also downgraded from 2.1 to 0.5 as 0.5 is where
most code in tutorials is written as well as in AI
3. learnt that pip install uses Python Package Index (PyPI) offical webpage as repo for install libraries, however pytorch has its own
repo on its webpage as pip has limit size for GB and big files so we pass index_url to access private repo or something outside PyPI offical repo
with all packages
4. smoke test is working!
5. learnt that train_envs = DummyVectorEnv([get_env for _ in range(8)]), creates 8 envs to run in parallel during training
6. `shared_ppo_policy = PPOPolicy(
    actor=actor,
    critic=critic,
    optim=optim,
    dist_fn=lambda loc, scale: torch.distributions.Normal(loc, scale),
    action_space=env_tmp.action_space,
    action_scaling=True,
    action_bound_method="clip"
)`
this code has shared_popo_policy aka MAPPO, actor, critic, optim, and dist_fn is saying given mean [loc] and standard deviation [how far are data spread out in dist] take them and torch.distrubtions.normal function returns single value, since its continous action space problem, we return
a single number randomly from distribution created, as our actions are continous,
action_scaling=True, neural networks work best [+1 to -1] for output why? not sure need to research but it struggles for anything beyond or less than that unless we ask it to scale to distrubtion of actions which can be +5 to -5, so it works well, last one if we get say 2.1 but we are allowed to use 2.0 only for action, we dont crash env we just downgrade it to 2.0 , clip it !
7. CUDA is working!

# 10/6/2026

1. Created the `analysis/` folder and added notes on training parameters and network output/action scaling.
2. Added a Hydra YAML config and confirmed that the training script loads it. Training settings such as `max_epoch`, `step_per_epoch`, `step_per_collect`, `repeat_per_collect`, `batch_size`, and `episode_per_test` now come from `cfg.training` instead of being hard-coded in the trainer.
3. Used Hydra's runtime output directory to organize run artifacts. Added periodic model checkpoints containing the policy state, optimizer state, epoch, and environment step; the save interval is read from `cfg.training.save_model_interval`.
4. Integrated deterministic evaluation-video recording into checkpoint saves. The evaluation video is generated with the MultiWalker parallel environment in `rgb_array` mode and written as an MP4 in the Hydra run directory.
5. Set up TensorBoard logging with a `SummaryWriter` writing under the run's `tb_logs` directory, and passed the logger to the Tianshou trainer.

# 10/7/2026

1. Finished setting up the Hydra training configuration and ran the configured training/evaluation workflow.
2. Installed TensorBoard and confirmed the run wrote TensorBoard event data to its `tb_logs/` directory.
3. Evaluation-video recording is running from the training checkpoint callback. The run created one MP4 per epoch.
4. Saved model and optimizer checkpoints for epochs 1–5. The run artifacts are under `outputs/2026-10-07/00-29-11/` and include the saved Hydra config, training log, five checkpoints, five evaluation videos, and a TensorBoard event file.
5. Checked the run configuration: this is still a smoke-scale run (5 epochs, 1,000 steps per epoch, batch size 32, and 1 test episode). Treat these artifacts as an end-to-end pipeline check, not a completed/fully trained model.

# 10/8/2026

1. Wrote `analysis/tianshou_framework.md` to document how the Tianshou framework fits into the project.
2. Organized the training output format so run artifacts are grouped in the output directory, including checkpoints, evaluation videos, reward graphs, and TensorBoard logs.
3. Launched TensorBoard using the run's `tb_logs/` directory and confirmed that it works. I do not understand all the information in the dashboard yet, so interpreting the charts and metrics is still something to learn.
4. Explored video-encoding options that may make video generation faster and use less CPU, at the cost of larger video files. This is still an approach under consideration, not a confirmed performance improvement.
5. Generated team-reward graphs for training and evaluation and saved them with the checkpoint outputs.

# 10/11/2026

1. Added the MultiWalker environment and reward settings to the Hydra config, and wired them into both training and evaluation environments so reward-related parameters can be adjusted in one place.
2. Wrote `analysis/reward_function.md` to summarize the MultiWalker reward parameters, their effects, and how shared rewards work.
3. Wrote `analysis/train_parameters.md` to explain the training hyperparameters, including how collection frequency, update repeats, and batch size affect training.
4. Learned how to use TensorBoard to monitor training and what to look for in the reward and episode-length charts, including how reward and episode length together can indicate whether the walkers are progressing or merely surviving. Documented these notes in `analysis/tensorboard.md`.

# 10/12/2026

1. Started the experimental phase properly: I ran a first set of training experiments and began comparing the effects of reward tuning on agent behavior.
2. I reviewed both Experiment One and Experiment Two in detail before moving forward. They showed a clear pattern: the agent learns survival and balancing more than pure forward motion, but it often shifts between aggressive movement and cautious balancing depending on the penalty strength.
3. In Experiment One, the default reward settings (`fall_reward = -10`) produced unstable learning. Epoch 2 and later epochs showed a mix of improved rewards and the agent sometimes switching to aggressive forward motion without keeping balance. The strongest behavior looked like the model was trying to balance and survive rather than chase reward aggressively.
4. In Experiment Two, I increased the penalty for falling to `fall_reward = -15` and kept the same training loop. The stronger penalty helped the policy prefer staying upright, and by Epoch 4 and Epoch 5 the agent showed noticeably more stable behavior and better reward trends than earlier runs. The middle walker still tended to die first, but the overall policy was more consistent.
5. Current takeaway: the agent is learning, but it is still under-optimizing the trade-off between staying alive and moving forward. The stronger fall penalty is helping, so the next step is to keep increasing the penalty to push the policy toward safer balancing behavior rather than reckless forward motion.
6. I have now moved from setup and understanding into real experimentation, and I am using the run outputs, videos, and reward plots as the evidence base for the next tuning cycle.
7. The current training outputs are in `outputs/2026-10-11/` and `outputs/2026-10-12/`, and I am using those checkpoints to compare how reward changes affect training progression over time.

# 10/13/2026

1. Wrote a PPO summary in `documents/ppo_explained.md`, covering the actor and critic, rollout collection, advantage estimates, mini-batch updates, clipping, on-policy learning, and entropy.
2. Read about additional PPO training metrics and how to interpret their TensorBoard graphs: approximate KL divergence, policy loss, value loss, and explained variance. These metrics can help diagnose how much the policy changes during updates, how the actor and critic losses behave, and how well the critic predicts returns.
3. Next: investigate how to expose and log these metrics from Tianshou in TensorBoard, then add them to the training workflow so future runs can be monitored alongside reward and episode length.

