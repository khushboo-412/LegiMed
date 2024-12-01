import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import pandas as pd
from tqdm.auto import tqdm  # Import tqdm for progress bar visualization

# Set up CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the BioBART model and tokenizer
model_name = "GanjinZero/biobart-large"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
model.eval()  # Set the model to evaluation mode

# Load the CSV file
df = pd.read_csv('reports.csv')
print("Loaded data with", len(df), "entries")

# Define a function to generate a summary for a single text
def generate_summary(input_text):
    prompt = f"summarize: {input_text}"

    inputs = tokenizer.encode(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=2048)
    inputs = inputs.to(device)
    outputs = model.generate(
        inputs,
        max_length=1024, 
        num_beams=4,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

tqdm.pandas(desc="Generating summaries")  # Initialize tqdm within pandas
df['SummaryBioBART'] = df['REPORT'].progress_apply(generate_summary)

# Save the updated DataFrame back to the same file
df.to_csv('reports.csv', index=False)
