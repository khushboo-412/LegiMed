import torch
from datasets import load_dataset
import evaluate
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from tqdm.auto import tqdm  # Import tqdm for progress bar
import pandas as pd  

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  


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

# Load the T5-small model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)  # Move model to CUDA device

# Preprocess function for the dataset
def preprocess_function(examples):
    inputs = ["Report: " + report for report in examples['input']]
    targets = ["Summary: " + summary for summary in examples['output']]
    
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
        labels = {k: torch.tensor(v).to(device) for k, v in labels.items()}  # Move labels to CUDA device

    model_inputs["labels"] = labels["input_ids"]
    return {k: torch.tensor(v).to(device) for k, v in model_inputs.items()}  # Move inputs to CUDA device

# Preprocess the dataset
train_dataset = train_data.map(preprocess_function, batched=True)
test_dataset = test_data.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    warmup_steps=500,
    logging_steps=100,
    save_total_limit=3,
    save_strategy="epoch",
    gradient_accumulation_steps=2,
    #fp16=True,  
    dataloader_num_workers=4,
    load_best_model_at_end=True 
)

def compute_rouge_and_save_csv(test_data, csv_filename="t5.csv"):
    all_preds = []
    all_labels = []
    all_inputs = []

    progress_bar = tqdm(test_data, desc="Evaluating", leave=False)

    for test_example in progress_bar:
        input_text = test_example['input']
        reference_summary = test_example['output']  
        generated_summary = generate_summary(input_text)
        print(generated_summary)

        all_preds.append(generated_summary)
        all_labels.append(reference_summary)
        all_inputs.append(input_text)  
    rouge_result = rouge.compute(predictions=all_preds, references=all_labels)

    results_df = pd.DataFrame({
        "Input": all_inputs,
        "Reference Summary": all_labels,
        "Generated Summary": all_preds
    })
    results_df.to_csv(csv_filename, index=False)  
    print(f"Results saved to {csv_filename}")

    return rouge_result

def generate_summary(input_text):
    prompt = f"""
    Generate a concise summary highlighting legal implications and key points for a lawyer. 
    Focus on clinical decisions, medication and dosage details, lab reports, treatment options, 
    progress notes, and expert consultations. Consider the legal relevance of each point: {input_text}
    """
    #prompt = f"Generate a concise summary highlighting legal implications and key points for a lawyer. Summarize the following clinical text based on key factors: Clinical decisions made, including dates and reasons, Medications and dosages, Lab reports and diagnostic results, Treatment options accepted or rejected, Progress and follow-up notes, Expert consultations and reports: {input_text}"

    #prompt = "Report: " + input_text  
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)  # Use device for inference
    outputs = model.generate(
        inputs, 
        max_length=1000,  
        min_length=100,   
        num_beams=4,  
        early_stopping=True,
        length_penalty=1.0,  
        no_repeat_ngram_size=3  
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)
trainer.train()

eval_results = trainer.evaluate()
print(eval_results)

rouge_scores = compute_rouge_and_save_csv(test_data)
print("ROUGE Scores:", rouge_scores)


'''
Output:

/usr/bin/python3 /cronus_data/ksingh/run/t5small_final.py
19756
15804
3952
/usr/local/lib/python3.8/dist-packages/huggingface_hub/file_download.py:1150: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
You are using the default legacy behaviour of the <class 'transformers.models.t5.tokenization_t5.T5Tokenizer'>. If you see this, DO NOT PANIC! This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thouroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565
Map:   0%|                                                                                                                               | 0/15804 [00:00<?, ? examples/s]/usr/local/lib/python3.8/dist-packages/transformers/tokenization_utils_base.py:3660: UserWarning: `as_target_tokenizer` is deprecated and will be removed in v5 of Transformers. You can tokenize your labels by using the argument `text_target` of the regular `__call__` method (either in the same call as your input texts if you use the same keyword arguments, or in a separate call.
  warnings.warn(
/cronus_data/ksingh/run/t5small_final.py:46: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  return {k: torch.tensor(v).to(device) for k, v in model_inputs.items()}  # Move inputs to CUDA device
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15804/15804 [00:39<00:00, 404.00 examples/s]
Map: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3952/3952 [00:09<00:00, 401.85 examples/s]
/usr/local/lib/python3.8/dist-packages/accelerate/accelerator.py:451: FutureWarning: Passing the following arguments to `Accelerator` is deprecated and will be removed in version 1.0 of Accelerate: dict_keys(['dispatch_batches']). Please pass an `accelerate.DataLoaderConfiguration` instead: 
dataloader_config = DataLoaderConfiguration(dispatch_batches=None)
  warnings.warn(
Detected kernel version 5.4.0, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.
  0%|                                                                                                                                            | 0/1482 [00:00<?, ?it/s]/usr/local/lib/python3.8/dist-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
  warnings.warn('Was asked to gather along dimension 0, but all '
{'eval_loss': 0.993397057056427, 'eval_runtime': 25.6721, 'eval_samples_per_second': 153.941, 'eval_steps_per_second': 4.83, 'epoch': 1.0}                                
{'loss': 1.4735, 'learning_rate': 1.3252361673414307e-05, 'epoch': 1.01}                                                                                                  
 34%|███████████████████████████████████████████▊                                                                                      | 500/1482 [03:06<26:00,  1.59s/it]/usr/local/lib/python3.8/dist-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
  warnings.warn('Was asked to gather along dimension 0, but all '
{'eval_loss': 0.9510758519172668, 'eval_runtime': 25.7197, 'eval_samples_per_second': 153.656, 'eval_steps_per_second': 4.821, 'epoch': 2.0}                              
{'loss': 1.0875, 'learning_rate': 6.504723346828611e-06, 'epoch': 2.02}                                                                                                   
 67%|███████████████████████████████████████████████████████████████████████████████████████                                          | 1000/1482 [06:02<03:36,  2.23it/s]/usr/local/lib/python3.8/dist-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
  warnings.warn('Was asked to gather along dimension 0, but all '
{'eval_loss': 0.939447820186615, 'eval_runtime': 25.6007, 'eval_samples_per_second': 154.371, 'eval_steps_per_second': 4.844, 'epoch': 3.0}                               
{'train_runtime': 533.8836, 'train_samples_per_second': 88.806, 'train_steps_per_second': 2.776, 'train_loss': 1.207253810043438, 'epoch': 3.0}                           
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1482/1482 [08:53<00:00,  2.78it/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 124/124 [00:24<00:00,  4.99it/s]
{'eval_loss': 0.939447820186615, 'eval_runtime': 25.0459, 'eval_samples_per_second': 157.79, 'eval_steps_per_second': 4.951, 'epoch': 3.0}
ROUGE Scores: {'rouge1': 0.49494984289478894, 'rouge2': 0.30802189050167483, 'rougeL': 0.3833390778303636, 'rougeLsum': 0.3835979122955222} 
'''



