#!/usr/bin/env bash
set -e
python main.py \
  --algorithm fedbeat \
  --dataset mnist \
  --partition noniid \
  --model cnn \
  --rounds 10 --clients-per-round 10 \
  --config-overrides '{
    "fedbeat": {
      "local_epochs_backbone": 1,
      "local_epochs_posterior": 1,
      "warmup_rounds": 2,
      "dbfat": {"beta": 1.5, "pgd_steps_probe": 5},
      "attack_train": {"type":"fgsm","eps":0.3,"steps":1,"prob":1.0},
      "attack_eval": {"enabled": true, "types": ["fgsm","square"], "square": {"eps":0.3,"queries":300}},
      "posterior": {"ensemble": 5, "freeze_backbone_during_posterior": true}
    }
  }'
