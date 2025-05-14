"""
Utility functions for the knowledge distillation pipeline.
"""

import os
import torch


def get_available_device():
    """
    Get the best available device for training.
    
    Returns:
        String representing the device to use
    """
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        print(f"🚀 Using GPU: {device_name}")
    elif torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon
        print(f"🚀 Using Apple Silicon MPS")
    else:
        device = "cpu"
        print(f"⚠️ No GPU available, using CPU")
    
    return device

def print_model_size_comparison(teacher_model, student_model):
    """
    Print size comparison between teacher and student models.
    
    Args:
        teacher_model: The teacher model
        student_model: The student model
    """
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    teacher_params = count_parameters(teacher_model)
    student_params = count_parameters(student_model)
    reduction = (1 - student_params / teacher_params) * 100
    
    print(f"📊 Model size comparison:")
    print(f"   - Teacher: {teacher_params:,} parameters")
    print(f"   - Student: {student_params:,} parameters")
    print(f"   - Reduction: {reduction:.2f}%")