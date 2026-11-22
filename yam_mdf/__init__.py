"""MDF-lite: multimodal diffusion forcing (independent noise per time step and modality) on the YAM screw task.

  cache.py   LeRobot dataset -> per-episode feature cache (10 Hz; frozen ResNet18 image features)
  data.py    modalities, normalisation, PCA of image features, training windows
  model.py   causal transformer denoiser over (time x modality) tokens, noise schedule, DDIM sampling
  train.py   training + held-out evaluation (action error, robustness to noisy observations)
  policy.py  load a checkpoint and produce action chunks from an observation history (for deployment)
"""
