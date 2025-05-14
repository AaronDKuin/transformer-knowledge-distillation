"""
Model loading and configuration utilities.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer


def verify_tokenizer_compatibility(teacher_id, student_id):
    """
    Verify that student and teacher tokenizers produce compatible outputs.
    
    Args:
        teacher_id: The teacher model ID
        student_id: The student model ID
    """

    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_id)
    student_tokenizer = AutoTokenizer.from_pretrained(student_id)
    
    # Sample input for testing
    sample = "This is a basic example, with different words to test."
    
    # Test tokenization compatibility
    try:
        assert teacher_tokenizer(sample) == student_tokenizer(sample), "Tokenizers haven't created the same output"
        print("✅ Tokenizers are compatible")
    except AssertionError as e:
        print(f"⚠️ Warning: {e}")
        print("Tokenization differences may affect distillation quality.")

def load_models(teacher_id, student_id, num_labels, id2label, label2id):
    """
    Load teacher and student models for distillation.
    
    Args:
        teacher_id: The teacher model ID
        student_id: The student model ID
        num_labels: Number of classification labels
        id2label: Mapping from ID to label
        label2id: Mapping from label to ID
    
    Returns:
        teacher_model, student_model
    """
    # Load teacher model
    teacher_model = AutoModelForSequenceClassification.from_pretrained(
        teacher_id,
        num_labels=num_labels, 
        id2label=id2label,
        label2id=label2id,
    )
    
    # Load student model
    student_model = AutoModelForSequenceClassification.from_pretrained(
        student_id,
        num_labels=num_labels, 
        id2label=id2label,
        label2id=label2id,
    )
    
    return teacher_model, student_model