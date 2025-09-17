# A Pytorch implementation of our paper ADAGRAD-FUSION: ADAPTIVE GRADIENT FUSION FOR MEMORY-EFFICIENT ECG FOUNDATION MODEL FINE-TUNING
# Databases and Backbones
* Four downstream datasets: 
  The Chapman-Shaoxing dataset, the PTB-XL dataset, the Ningbo dataset, and the G12EC dataset.
* Quick download the four downsteam datasets: 
  wget -r -N -c -np https://physionet.org/files/challenge-2021/1.0.3/
* Teacher Model Pre-training Dataset:
  The teacher model was pre-trained on the CODE-15 dataset, which can be downloaded from:https://zenodo.org/records/4916206
* Pre-trained Backbones:
  The pre-trained backbones are available on [Hugging Face](https://huggingface.co/KAZABANA/Foundation-Models-for-ECG-classification/tree/main).
* Requirements: 
  All necessary packages and dependencies for this project can be installed by running:pip install -r requirements.txt
# Main Code
* datacollection: This module handles the initial data preprocessing, converting raw datasets into the .hdf5 format and then performing data splitting.
* main: The primary entry point for running experiments. This is where you can adjust various parameters.
* pipeline_ecg: Contains the implementation for knowledge distillation, including the definitions for both the student and teacher models.
* pipeline_pretrain: Includes the code related to the pre-training process.
* Half_Trainer: Manages gradient-related operations.
* evaluation: This module is used for processing and evaluating the experimental results.
* model_src_ecg: Defines the model architectures, including the default model (model_code_default) and the LoRA layers (Lora_layer_default).
* gradient_pruning: Provides tools for implementing gradient sparsification.
# Hyper-parameter
* learning_rate: The learning rate for the models, which can be adjusted in main.py, pipeline_ecg, and pipeline_pretrain.py.
Teacher model: 0.0025
Student model: 0.002
* zo_eps: The perturbation step size, which can be configured in main.py. Recommended values are between 1e-3 and 1e-4.
* finetune_label_radio: The ratio for splitting the data into training and testing sets, adjustable in main.py.
* r: The rank for the LoRA layers, which can be set in main.py. Suggested values are 4, 8, or 16.
* bp_batch: The batch size for backpropagation, configurable in main.py. Recommended values are 2, 4, or 8.
* coef: This parameter dynamically adjusts the balance between zero-order and first-order optimization in the Half_Trainer. It can also be set in main.py. Optimal values may vary across different datasets. Suggested values: 0.85, 0.90, 0.95, 0.99.
* positive_probably: The threshold for the binary classification auxiliary head, which should correspond to the proportion of positive samples in each dataset.
* aux_weight: The weight of the auxiliary head's loss in the total loss function during training. This should be set to 0 during inference. Optimal values may differ depending on the dataset. Suggested values: 0.05, 0.2, 0.8, 1.
# Directory Structure
* Preprocessed_dataset: This directory stores the preprocessed datasets in .hdf5 format.
* pretrained_checkout: Checkpoints for the models are saved in this directory.
* result: .The experimental results are saved as .npy files in this directory.

