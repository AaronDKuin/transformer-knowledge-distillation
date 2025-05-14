"""
Custom trainer implementation for knowledge distillation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrainingArguments, Trainer
from evaluate import load
import numpy as np

class DistillationTrainingArguments(TrainingArguments):
    """
    Custom training arguments for distillation.
    """
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.alpha = alpha
        self.temperature = temperature
        
class DistillationTrainer(Trainer):
    """
    Custom trainer for knowledge distillation from teacher to student.
    """
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        # Place teacher on same device as student
        self._move_model_to_device(self.teacher, self.model.device)
        self.teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss function that combines:
        - Regular student loss
        - KL divergence between teacher and student logits
        """
        # Compute student output
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss
        
        # Compute teacher output
        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs)
        
        # Assert size
        assert outputs_student.logits.size() == outputs_teacher.logits.size()
        
        # Soften probabilities and compute distillation loss
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (loss_function(
            F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
            F.softmax(outputs_teacher.logits / self.args.temperature, dim=-1)) * (self.args.temperature ** 2))
        
        # Return weighted student loss
        loss = self.args.alpha * student_loss + (1. - self.args.alpha) * loss_logits
        return (loss, outputs_student) if return_outputs else loss

def create_training_args(config, distillation_config, output_dir):
    """
    Create training arguments for distillation.
    
    Args:
        config: Training configuration dictionary
        distillation_config: Distillation configuration dictionary
        output_dir: Output directory for saving models
        
    Returns:
        DistillationTrainingArguments object
    """
    # Create training arguments
    training_args = DistillationTrainingArguments(
        output_dir=output_dir,
        alpha=distillation_config["alpha"],
        temperature=distillation_config["temperature"],
        **config
    )
    
    return training_args

def setup_trainer(
    student_model, 
    teacher_model, 
    tokenized_datasets, 
    training_args, 
    data_collator, 
    tokenizer
):
    """
    Set up the distillation trainer.
    
    Args:
        student_model: The student model to train
        teacher_model: The teacher model to learn from
        tokenized_datasets: Processed datasets
        training_args: Training arguments
        data_collator: Data collator for batching
        tokenizer: Tokenizer
        
    Returns:
        Configured DistillationTrainer
    """
    # Set up metrics for evaluation
    accuracy_metric = load("accuracy")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        acc = accuracy_metric.compute(predictions=predictions, references=labels)
        return {
            "accuracy": acc["accuracy"],
        }
    
    # Create and return the trainer
    trainer = DistillationTrainer(
        student_model,
        training_args,
        teacher_model=teacher_model,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    return trainer