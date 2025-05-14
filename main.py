"""
Main entry point for the knowledge distillation pipeline.
Run this script to perform knowledge distillation from teacher to student model.
"""

import argparse
import os
from config import MODEL_CONFIG, DATASET_CONFIG, TRAINING_CONFIG, DISTILLATION_CONFIG
from data import load_and_process_dataset
from models import verify_tokenizer_compatibility, load_models
from trainer import create_training_args, setup_trainer
from utils import get_available_device, print_model_size_comparison

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Knowledge Distillation for Transformers")
    
    # Model arguments
    parser.add_argument("--teacher_id", type=str, default=MODEL_CONFIG["teacher_id"],
                        help="HuggingFace model ID for teacher")
    parser.add_argument("--student_id", type=str, default=MODEL_CONFIG["student_id"],
                        help="HuggingFace model ID for student")
    parser.add_argument("--repo_name", type=str, default=MODEL_CONFIG["repo_name"],
                        help="Repository name for saving model")
    
    # Dataset arguments
    parser.add_argument("--dataset_id", type=str, default=DATASET_CONFIG["dataset_id"],
                        help="HuggingFace dataset ID")
    parser.add_argument("--dataset_config", type=str, default=DATASET_CONFIG["dataset_config"],
                        help="Dataset configuration")
    parser.add_argument("--max_length", type=int, default=DATASET_CONFIG["max_length"],
                        help="Maximum sequence length")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=TRAINING_CONFIG["num_train_epochs"],
                        help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=TRAINING_CONFIG["per_device_train_batch_size"],
                        help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=TRAINING_CONFIG["per_device_eval_batch_size"],
                        help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=TRAINING_CONFIG["learning_rate"],
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=TRAINING_CONFIG["seed"],
                        help="Random seed")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    
    # Distillation arguments
    parser.add_argument("--alpha", type=float, default=DISTILLATION_CONFIG["alpha"],
                        help="Weight for original loss vs distillation loss")
    parser.add_argument("--temperature", type=float, default=DISTILLATION_CONFIG["temperature"],
                        help="Temperature for softening probability distributions")
    
    # HuggingFace Hub arguments
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to HuggingFace Hub")
    parser.add_argument("--hub_token", type=str, default=None, help="HuggingFace Hub token")
    
    return parser.parse_args()

def main():
    """Main function to run the knowledge distillation pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Configure model paths
    teacher_id = args.teacher_id
    student_id = args.student_id
    repo_name = args.repo_name
    
    # Create output directory
    os.makedirs(repo_name, exist_ok=True)
        
    # Verify tokenizer compatibility
    verify_tokenizer_compatibility(teacher_id, student_id)
    
    # Load and process dataset
    print(f"Loading and processing dataset: {args.dataset_id}/{args.dataset_config}")
    tokenized_datasets, data_collator, tokenizer, num_labels, label2id, id2label = load_and_process_dataset(
        args.dataset_id, args.dataset_config, teacher_id, args.max_length
    )
    
    # Load models
    print(f"Loading teacher model: {teacher_id}")
    print(f"Loading student model: {student_id}")
    teacher_model, student_model = load_models(
        teacher_id, student_id, num_labels, id2label, label2id
    )
    
    # Print model size comparison
    print_model_size_comparison(teacher_model, student_model)
    
    # Create training arguments
    training_config = TRAINING_CONFIG.copy()
    training_config.update({
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "fp16": args.fp16,
    })
    
    distillation_config = {
        "alpha": args.alpha,
        "temperature": args.temperature,
    }
    
    # Create training arguments
    training_args = create_training_args(training_config, distillation_config, repo_name)
    
    # Setup the trainer
    trainer = setup_trainer(
        student_model,
        teacher_model,
        tokenized_datasets,
        training_args,
        data_collator,
        tokenizer
    )
    
    # Start training
    print(f"Starting knowledge distillation training for {args.epochs} epochs")
    trainer.train()
    
    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    # Save the final model
    trainer.save_model(repo_name)
    print(f"Model saved to {repo_name}")
    


if __name__ == "__main__":
    main()