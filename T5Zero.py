import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
from tqdm.auto import tqdm

# Set up CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "t5-small"  # You can replace this with a different model name if needed
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
model.eval()  # Set the model to evaluation mode

# Load the few-shot examples CSV
few_shot_csv_path = '/home/ksingh/ams/few_shot.csv'
few_shot_df = pd.read_csv(few_shot_csv_path)

# Generate the few-shot examples for the prompt
few_shot_examples = ""
for _, row in few_shot_df.iterrows():
    few_shot_examples += f"Summary: {row['ChatGPT 3.5']}\n\n"

# Load the main dataset
main_csv_path = '/home/ksingh/ams/reports_with_few_shots.csv'
df = pd.read_csv(main_csv_path)
print("Loaded data with", len(df), "entries")

# Define a function to generate summaries
def generate_summary(input_text):
    prompt = f"""
    summarize for legal review:  {input_text}
    Summary:"""

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=10000).to(device)
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=10000,  # Adjust max length if needed
        min_length=100,   
        num_beams=4,  
        early_stopping=True,
        length_penalty=1.0,  
        no_repeat_ngram_size=3 
    )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(summary)
    return summary

# Generate summaries for the reports in the dataframe
tqdm.pandas(desc="Generating summaries")
df['SummaryT5few'] = df['REPORT'].progress_apply(generate_summary)

# Save the updated DataFrame back to the same file
df.to_csv('/home/ksingh/ams/reports_with_few_shots.csv', index=False)
print("Updated DataFrame saved with summaries.")
