# Mean Flow Matching Implementation Details
Our code origins from two repo:
- [flow matching](https://github.com/facebookresearch/flow_matching)
- [Mean Flow](https://github.com/haidog-yaqub/MeanFlow)
## Pre-training (From Scratch)
Data: The original half-moon.
Model: Mean velocity MLP. The way I intergrate factor r into model is: 
1. Not Using time embedder just as the Mean Flow ([Mean Flow](https://github.com/haidog-yaqub/MeanFlow)).
2. Addation on t and r, just as the Mean Flow did.
3. Concat the new result of r with x. The main block of network stays as before.

Loss: Using new target $$ 

Step: 20K
Loss Curve:
Unstable.

## Lora-one reinitialization(Gradient Calculation)
Gradient: Using u(z_t, 0, 1) as loss, calculate lora embedding when set positional embedding fixed(or not?)

## Finetuning
Baseline: w/o mean flow pre-train, w/o mean flow finetune

Baseline1: w. mean flow pre-train, w. mean flow finetune

Our method: w. mean flow pre-train, w. LoRA-One, w. mean flow finetune