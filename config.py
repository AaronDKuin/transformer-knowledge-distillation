"""
Configuration file for knowledge distillation settings.
You can customize model paths, training parameters, and distillation settings here.
"""

# Model configuration
MODEL_CONFIG = {
    "student_id": "google/bert_uncased_L-2_H-128_A-2",
    "teacher_id": "textattack/bert-base-uncased-SST-2",
    "repo_name": "tiny-bert-sst2-distilled",
}

# Dataset configuration
DATASET_CONFIG = {
    "dataset_id": "glue",
    "dataset_config": "sst2",
    "max_length": 512,
}

# Training configuration
TRAINING_CONFIG = {
    "num_train_epochs": 7,
    "per_device_train_batch_size": 128,
    "per_device_eval_batch_size": 128,
    "fp16": True,
    "learning_rate": 6e-5,
    "seed": 42,
    "logging_strategy": "epoch",
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "save_total_limit": 2,
    "load_best_model_at_end": True,
    "metric_for_best_model": "accuracy",
    "report_to": "tensorboard",
}

# Distillation configuration
DISTILLATION_CONFIG = {
    "alpha": 0.5,  # Weight for original loss vs distillation loss
    "temperature": 4.0,  # Temperature for softening probability distributions
}
