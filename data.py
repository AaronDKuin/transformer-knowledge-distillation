"""
Data processing utilities for loading and preprocessing datasets.
"""

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

def load_and_process_dataset(dataset_id, dataset_config, tokenizer_id, max_length=512):
    """
    Load and process the dataset for training.
    
    Args:
        dataset_id: The HuggingFace dataset ID
        dataset_config: The specific configuration of the dataset
        tokenizer_id: The tokenizer to use for processing
        max_length: Maximum sequence length
    
    Returns:
        Processed dataset and data collator
    """
    # Load dataset
    dataset = load_dataset(dataset_id, dataset_config)
    print(f"Dataset loaded: {dataset}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    
    # Define processing function
    def process(examples):
        tokenized_inputs = tokenizer(
            examples["sentence"], truncation=True, max_length=max_length
        )
        return tokenized_inputs
    
    # Process dataset
    tokenized_datasets = dataset.map(process, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    
    # Create data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Create label mappings
    labels = tokenized_datasets["train"].features["labels"].names
    num_labels = len(labels)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
    
    return tokenized_datasets, data_collator, tokenizer, num_labels, label2id, id2label