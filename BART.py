import torch
from datasets import load_dataset
import evaluate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from tqdm.auto import tqdm  # Import tqdm for progress bar

# Load the ROUGE metric from the evaluate library
rouge = evaluate.load("rouge")

# Load the dataset
dataset = load_dataset("Technoculture/synthetic-clinical-notes-embedded")

# Filter dataset to include only rows where the 'task' column has value 'Summarization'
dataset_subset = dataset['train'].filter(lambda example: example['task'] == 'Summarization')


print(len(dataset_subset))

# Split the dataset into 80:20 ratio for training and testing
dataset_split = dataset_subset.train_test_split(test_size=0.2)
train_data = dataset_split['train']
test_data = dataset_split['test']

print(len(train_data))
print(len(test_data))

# Set up CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the facebook/bart-large-cnn model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)  # Move model to CUDA device

# Preprocess function for the dataset
def preprocess_function(examples):
    inputs = ["Report: " + report for report in examples['input']]
    targets = [summary for summary in examples['output']]  # BART model doesn't require "Summary:" prefix

    # Tokenize inputs and targets for BART
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    # Convert lists to tensors and move to the appropriate device
    model_inputs["labels"] = torch.tensor(labels["input_ids"]).to(device)
    return {k: torch.tensor(v).to(device) for k, v in model_inputs.items()}

# Preprocess the dataset
train_dataset = train_data.map(preprocess_function, batched=True)
test_dataset = test_data.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Function to compute ROUGE score for generated summaries
def compute_rouge(test_data):
    all_preds = []
    all_labels = []

    # Initialize the progress bar
    progress_bar = tqdm(test_data, desc="Evaluating", leave=False)

    for test_example in progress_bar:
        input_text = test_example['input']
        reference_summary = test_example['output']  # Ground truth summary

        # Generate summary using the model
        generated_summary = generate_summary(input_text)
        
        all_preds.append(generated_summary)
        all_labels.append(reference_summary)

    # Compute the ROUGE score
    rouge_result = rouge.compute(predictions=all_preds, references=all_labels)
    return rouge_result

# Function to generate summary using the model
def generate_summary(input_text):
    prompt = "Report: " + input_text  # Add "Report" prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)  # Use device for inference
    outputs = model.generate(
        inputs, 
        max_length=1000,  # Increased max length
        min_length=100,  # Minimum length
        num_beams=4,     # Beam search for better quality
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Set up the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(eval_results)

# Compute ROUGE score with progress bar
rouge_scores = compute_rouge(test_data)
print("ROUGE Scores:", rouge_scores)


'''
/usr/bin/python3 /cronus_data/ksingh/run/bart.py
19756
15804
3952
/usr/local/lib/python3.8/dist-packages/huggingface_hub/file_download.py:1150: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
Map:   0%|                                                                                            | 0/15804 [00:00<?, ? examples/s]/usr/local/lib/python3.8/dist-packages/transformers/tokenization_utils_base.py:3660: UserWarning: `as_target_tokenizer` is deprecated and will be removed in v5 of Transformers. You can tokenize your labels by using the argument `text_target` of the regular `__call__` method (either in the same call as your input texts if you use the same keyword arguments, or in a separate call.
  warnings.warn(
/cronus_data/ksingh/run/bart.py:47: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  return {k: torch.tensor(v).to(device) for k, v in model_inputs.items()}
Map: 100%|██████████████████████████████████████████████████████████████████████████████| 15804/15804 [00:10<00:00, 1551.60 examples/s]
Map: 100%|████████████████████████████████████████████████████████████████████████████████| 3952/3952 [00:02<00:00, 1724.99 examples/s]
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
/usr/local/lib/python3.8/dist-packages/accelerate/accelerator.py:451: FutureWarning: Passing the following arguments to `Accelerator` is deprecated and will be removed in version 1.0 of Accelerate: dict_keys(['dispatch_batches']). Please pass an `accelerate.DataLoaderConfiguration` instead: 
dataloader_config = DataLoaderConfiguration(dispatch_batches=None)
  warnings.warn(
Detected kernel version 5.4.0, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.
  0%|                                                                                                         | 0/1482 [00:00<?, ?it/s/usr/local/lib/python3.8/dist-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
  warnings.warn('Was asked to gather along dimension 0, but all '
{'eval_loss': 0.5869863629341125, 'eval_runtime': 40.9815, 'eval_samples_per_second': 96.434, 'eval_steps_per_second': 3.026, 'epoch': 1.0}                                                                                                                                   
{'loss': 0.6822, 'learning_rate': 1.3252361673414307e-05, 'epoch': 1.01}                                                               
 34%|████████████████████████████████                                                               | 500/1482 [07:56<47:55,  2.93s/it]/usr/local/lib/python3.8/dist-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
  warnings.warn('Was asked to gather along dimension 0, but all '
{'eval_loss': 0.5607886910438538, 'eval_runtime': 40.9679, 'eval_samples_per_second': 96.466, 'eval_steps_per_second': 3.027, 'epoch': 2.0}                                                                                                                                   
{'loss': 0.5142, 'learning_rate': 6.504723346828611e-06, 'epoch': 2.02}                                                                
 67%|███████████████████████████████████████████████████████████████▍                              | 1000/1482 [15:54<08:43,  1.09s/it]/usr/local/lib/python3.8/dist-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
  warnings.warn('Was asked to gather along dimension 0, but all '
{'eval_loss': 0.5557055473327637, 'eval_runtime': 40.9261, 'eval_samples_per_second': 96.564, 'eval_steps_per_second': 3.03, 'epoch': 3.0}                                                                                                                                    
{'train_runtime': 1414.6207, 'train_samples_per_second': 33.516, 'train_steps_per_second': 1.048, 'train_loss': 0.5554157931473251, 'epoch': 3.0}                                                                                                                             
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 1482/1482 [23:34<00:00,  1.05it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████| 124/124 [00:40<00:00,  3.05it/s]
{'eval_loss': 0.5557055473327637, 'eval_runtime': 40.8609, 'eval_samples_per_second': 96.718, 'eval_steps_per_second': 3.035, 'epoch': 3.0}
ROUGE Scores: {'rouge1': 0.6002685671687338, 'rouge2': 0.417483288004898, 'rougeL': 0.48412670798033286, 'rougeLsum': 0.4842169618061882}
'''


