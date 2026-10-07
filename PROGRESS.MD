

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

## Still to do / follow-up

- Explain shared PPO policy versus `MultiAgentPolicyManager`, and PettingZoo Parallel versus AEC APIs and the Tianshou `PettingZooEnv` wrapper, including expected inputs and outputs.
- Document the evaluation-video path in detail, including CUDA inference, CPU/NumPy action conversion, batch shape, and video-frame handling.
- Add a Matplotlib helper to save team rewards, and persist reward history to JSON for later loading.
- Ask Gemini about possible evaluation-video performance optimizations.
- Launch TensorBoard and verify the UI can read the generated event file; installation and event-file creation are confirmed, but UI startup was not confirmed.
- Review the run log's 6,566 PettingZoo warnings about actions outside the `[-1, 1]` action space being clipped. The environment clips them, but action scaling/bounds should be checked before longer training.
- Note: `train/video_module.py` is a separate random-frame video example; the trainer's actual evaluation recording is implemented inside `train/training.py`.
