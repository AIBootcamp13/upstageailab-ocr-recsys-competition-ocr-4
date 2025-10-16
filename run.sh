#!/bin/bash

nohup uv run python wandb_sweep.py --sweep-id your_sweep_id --count 500 > sweep.log 2>&1 &
