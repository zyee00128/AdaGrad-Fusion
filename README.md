# ADAGRAD-FUSION

This repository contains a PyTorch implementation of the paper "ADAGRAD-FUSION: Adaptive Gradient Fusion for Memory-Efficient ECG Foundation Model Fine-tuning". It provides tools, training pipelines and utilities to fine-tune large ECG foundation models using a memory-efficient adaptive gradient fusion method.

## Key features

- Adaptive gradient fusion for memory-efficient fine-tuning
- Support for teacher-student knowledge distillation and LoRA-style adapters
- Gradient pruning / sparsification utilities
- Ready-to-run training and pretraining pipelines

## Requirements

This project targets Python 3.8+ and PyTorch 1.10+. Install required packages with:

```powershell
pip install -r requirements.txt
```

Refer to `requirements.txt` for exact package versions used in experiments.

## Quickstart

1. Prepare or download downstream datasets (examples used in the paper): Chapman-Shaoxing, PTB-XL, Ningbo, G12EC. The repository also uses CODE-15 for teacher pretraining.

Quick download the four downsteam datasets: 
```powershell
  wget -r -N -c -np https://physionet.org/files/challenge-2021/1.0.3/
```

3. The teacher model was pre-trained on the CODE-15 dataset, which can be downloaded from:
`https://zenodo.org/records/4916206`

4. The pre-trained backbones are available on [Hugging Face], which can be downloaded from:
`https://huggingface.co/KAZABANA/Foundation-Models-for-ECG-classification/tree/main`.

2. Preprocess raw data into the project's format with:

```powershell
python datacollection.py
```

3. Train or fine-tune a model (default settings) with:

```powershell
python main.py
```

4. Use `pipeline_pretrain.py` to run teacher pretraining, and `pipeline_ecg.py` for student distillation/fine-tuning workflows.

## Project structure

```
AdaGrad-Fusion/
├── datacollection.py        # data preprocessing and dataset splitting
├── main.py                 # main experiment entrypoint
├── pipeline_ecg.py         # distillation & fine-tuning pipeline
├── pipeline_pretrain.py    # pretraining pipeline for teacher models
├── Half_Trainer.py         # gradient fusion / optimizer logic
├── evaluation.py           # evaluation and metrics
├── helper_code.py          # utility helpers
├── model_src_ecg/          # model definitions and LoRA layers
├── gradient_pruning/       # gradient sparsification utilities
├── pretrained_checkpoint/  # example checkpoints (not all included)
└── result/                 # experiment outputs (.npy, logs)
```

## Important files and folders

- `datacollection.py` - converts raw ECG datasets into HDF5 and splits them for experiments
- `main.py` - run experiments and adjust hyperparameters
- `pipeline_ecg.py` - contains teacher-student training logic and evaluation hooks
- `pipeline_pretrain.py` - pretraining pipeline for the teacher
- `Half_Trainer.py` - implements the adaptive gradient fusion logic used in the paper
- `gradient_pruning/` - tools for sparsifying gradients when needed
- `model_src_ecg/` - model definitions, including LoRA layer implementations

## Hyper-parameters and tips

- learning_rate: Set in `main.py`, `pipeline_ecg.py`, and `pipeline_pretrain.py`. Common settings used in the paper:
  - Teacher model: 0.0025
  - Student model: 0.002
- zo_eps: Zero-order perturbation step size (recommended 1e-3 to 1e-4)
- finetune_label_ratio: Ratio for splitting labeled data for fine-tuning
- r: LoRA rank (typical choices 4, 8, 16)
- bp_batch: Backpropagation batch size (typical choices 2, 4, 8)
- coef: Fusion coefficient balancing zero-order and first-order updates (typical: 0.85–0.99)
- aux_weight: Weight for auxiliary binary head; set to 0 during inference

## Reproducing experiments

1. Install dependencies.
2. Prepare datasets and convert them with `datacollection.py`.
3. Optionally pretrain a teacher with `pipeline_pretrain.py` using CODE-15.
4. Run `main.py` or the tailored pipeline scripts to reproduce fine-tuning and evaluation.

## License & Citation

Please cite the ADAGRAD-FUSION paper when using this code in academic work. 

