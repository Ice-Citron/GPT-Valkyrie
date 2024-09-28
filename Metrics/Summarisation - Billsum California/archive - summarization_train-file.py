import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, concatenate_datasets, DatasetDict
import numpy as np
from rouge_score import rouge_scorer
import wandb
from transformers.integrations import WandbCallback


# load dataset
def load_billsum():
    """
    Load and combine the 'train' and 'test' splits of the Billsum dataset into a single 'train' set,
    and use 'ca_test' as the 'test' set.

    Returns:
        DatasetDict: A dictionary containing 'train' and 'test' datasets.
    """
    # Define the split names
    training_splits = ['train', 'test']
    test_split = 'ca_test'

    datasets_to_combine = []

    # Load and combine training splits
    for split in training_splits:
        try:
            ds = load_dataset("billsum", split=split)
            print(f"Loaded split: {split} with {len(ds)} examples.")
            datasets_to_combine.append(ds)
        except Exception as e:
            print(f"Could not load split '{split}': {e}")

    if not datasets_to_combine:
        raise ValueError("No training datasets were loaded. Please check the split names.")

    # Concatenate 'train' and 'test' splits into a single 'train' set
    combined_train = concatenate_datasets(datasets_to_combine)
    print(f"Combined train dataset size: {len(combined_train)} examples.")

    # Load the 'ca_test' split as the test set
    try:
        test_ds = load_dataset("billsum", split=test_split)
        print(f"Loaded test split: {test_split} with {len(test_ds)} examples.")
    except Exception as e:
        raise ValueError(f"Could not load test split '{test_split}': {e}")

    # Create a DatasetDict with 'train' and 'test' splits
    dataset_dict = DatasetDict({
        "train": combined_train,
        "test": test_ds
    })

    print(f"DatasetDict keys: {dataset_dict.keys()}")
    return dataset_dict


# Initialize tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


# Preprocess function
def preprocess_function(examples, tokenizer):
    """
    Tokenize the input texts and summaries.

    Args:
        examples (dict): A batch of examples from the dataset.
        tokenizer (GPT2TokenizerFast): The tokenizer to use.

    Returns:
        dict: Tokenized inputs and labels.
    """
    prefix = "summarize: "  # Prefix to guide the model for summarization

    # Prepare input texts with prefix
    inputs = [prefix + doc for doc in examples["text"]]

    # Tokenize inputs (source texts) with padding and truncation
    model_inputs = tokenizer(
        inputs,
        max_length=1024,            # Adjust based on your GPU's memory capacity
        truncation=True,
        padding="max_length"        # Uniform padding
    )

    # Tokenize labels (summaries) with padding and truncation
    labels = tokenizer(
        examples["summary"],
        max_length=128,             # Adjust as needed
        truncation=True,
        padding="max_length"        # Uniform padding
    )

    # Replace padding token IDs of the labels by -100 so they're ignored by the loss
    labels_ids = [
        [(token_id if token_id != tokenizer.pad_token_id else -100) for token_id in label]
        for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels_ids  # Assign processed labels

    return model_inputs


# Load and preprocess the dataset
dataset = load_billsum()
tokenized_datasets = dataset.map(                             # Preprocess the dataset using a lambda to pass the tokenizer
    lambda examples: preprocess_function(examples, tokenizer),
    batched=True,
    remove_columns=['text', 'summary', 'title']  # Remove original text columns to save memory
)
# Limit the test set to 100 examples
tokenized_datasets["test"] = tokenized_datasets["test"].shuffle(seed=42).select(range(100))
print(f"Limited test dataset size: {len(tokenized_datasets['test'])} examples.")


# Function to freeze layers based on variant type
def freeze_layers(model, variant_type):
    if variant_type == "noNorm":
        for name, param in model.named_parameters():
            if "ln" in name:
                param.requires_grad = False
    elif variant_type == "AttnOnly":
        for name, param in model.named_parameters():
            if "ln_2" in name:  # Freeze FFN layer norm
                param.requires_grad = False
    elif variant_type == "FFNonly":
        for name, param in model.named_parameters():
            if "ln_1" in name:  # Freeze attention layer norm
                param.requires_grad = False
    # For baseModel, we don't freeze any layers




import evaluate  # Import the evaluate library

# Initialize ROUGE and BLEU metrics
rouge = evaluate.load("rouge")
# bleu = evaluate.load("bleu")


import sacrebleu

def compute_metrics(eval_pred):
    """
    Compute ROUGE and BLEU metrics for summarization using SacreBLEU with smoothing.

    Args:
        eval_pred (EvalPrediction): Contains predictions and label_ids.

    Returns:
        dict: Average ROUGE and BLEU scores.
    """
    predictions, labels = eval_pred

    # Convert logits to token IDs by taking the argmax over the vocabulary dimension
    pred_ids = np.argmax(predictions, axis=-1)

    # Decode the predicted token IDs to text
    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    # Replace -100 in the labels with the pad token ID and decode
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Ensure that the predictions and references are lists of strings
    # and remove any leading/trailing whitespace
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # Compute ROUGE scores using the evaluate library
    rouge_result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )

    # Compute BLEU scores using SacreBLEU with smoothing
    bleu_scores = sacrebleu.corpus_bleu(
        decoded_preds,
        [decoded_labels],
        smooth_method='exp',       # Exponential smoothing
        smooth_value=0.1,
        force=True,                # Force compute even if length mismatch
        lowercase=True,            # Normalize case
        tokenize='13a'             # Tokenizer type (SacreBLEU default)
    )

    bleu_score = bleu_scores.score  # SacreBLEU returns a score attribute

    # Aggregate the results
    result = {
        "rouge1": rouge_result["rouge1"],
        "rouge2": rouge_result["rouge2"],
        "rougeL": rouge_result["rougeL"],
        "bleu": bleu_score
    }

    # Optional: Calculate average prediction length
    prediction_lens = [len(pred.split()) for pred in decoded_preds]
    result["gen_len"] = np.mean(prediction_lens)

    # Round the results to four decimal places for readability
    result = {k: round(v, 4) for k, v in result.items()}

    return result




# Fine-tuning function
def fine_tune_model(model, tokenizer, dataset, output_dir, variant, norm_type):
    # Initialize wandb run
    wandb.init(project=f"GPT-Valkyrie_{norm_type}-124m__{variant}__Billsum", reinit=True)
    run_name = wandb.run.name

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=20,
        per_device_eval_batch_size=20,
        warmup_steps=350,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=1,
        save_strategy="steps",
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000, # decided to just not do this at all
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        report_to="wandb",
        run_name=run_name,
        save_total_limit=5,  # Limit the total number of checkpoints
        fp16=True,                           # Enable mixed precision for memory efficiency
    )

    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal language modeling
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"], # even tho its meant to be called "validation" instead
        tokenizer=tokenizer,
        data_collator=data_collator,  # Use the updated data collator
        compute_metrics=compute_metrics,
        callbacks=[WandbCallback()],
    )

    trainer.train()
    wandb.finish()
    return trainer.model, run_name




# Main training loop
variants = ["AttnOnly", "FFNonly"] # ["baseModel", "noNorm", "AttnOnly", "FFNonly"]
norm_types = ["RMSN"] # ["LN", "RMSN"]

for norm_type in norm_types:        
    for variant in variants:
        print(f"Processing {norm_type} {variant} model...")

        # Use the correct base model for each variant
        model_path = f"shng2025/GPT-Valkyrie_{norm_type}-124m__{variant}__"
        model = GPT2LMHeadModel.from_pretrained(model_path)

        model.config.pad_token_id = tokenizer.pad_token_id
        # Print to verify
        print(f"Tokenizer pad token: {tokenizer.pad_token}")
        print(f"Tokenizer pad token ID: {tokenizer.pad_token_id}")
        print(f"Model pad token ID: {model.config.pad_token_id}")

        freeze_layers(model, variant)

        output_dir = f"./results/{norm_type}/{variant}"
        fine_tuned_model, run_name = fine_tune_model(model, tokenizer, tokenized_datasets, output_dir, variant, norm_type)

        # Save the model locally
        local_save_dir = f"./local_models/GPT-Valkyrie_{norm_type}-124m__{variant}__Billsum"
        fine_tuned_model.save_pretrained(local_save_dir)
        tokenizer.save_pretrained(local_save_dir)
        print(f"Model saved locally to {local_save_dir}")

        # Push the model to your HuggingFace Hub repository
        new_repo_name = f"shng2025/GPT-Valkyrie_{norm_type}-124m__{variant}__Billsum"
        fine_tuned_model.push_to_hub(new_repo_name, branch=run_name)
        tokenizer.push_to_hub(new_repo_name, branch=run_name)
        print(f"Model pushed to HuggingFace Hub: {new_repo_name}, branch: {run_name}")

print("Training completed for all variants and normalization types.")