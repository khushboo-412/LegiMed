from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
from tqdm import tqdm

# Setup tqdm for pandas progress bar
tqdm.pandas()

# Load the few-shot examples CSV
few_shot_csv_path = 'few_shot.csv'
few_shot_df = pd.read_csv(few_shot_csv_path)

# Generate the few-shot examples for the prompt
few_shot_examples = ""
for _, row in few_shot_df.iterrows():
    few_shot_examples += f"Example: {row['ChatGPT 3.5']}\n\n"

# Load the model and tokenizer onto the GPU
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="balanced")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="balanced")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set for padding handling

model.eval()

# Function to summarize medical documents with Chain-of-Thought prompting
def summarize_medical_document(document):
    input_string = f"""
    You are a helpful assistant trained to process medical documentation for legal review.
    Few-Shot Examples: {few_shot_examples} 
    Here is a medical document: {document}
    Analyze the document and provide a step-by-step explanation of key findings, treatments, and any notable outcomes. 
    First, identify the important details in the document. Then explain their significance or legal relevance. 
    Finally, provide a concise summary based on this analysis, including key medical events, treatments, and notable outcomes.
    This summary should be suitable for legal professionals looking at potential litigation or case studies.
    Finish your response with 'Summary: ' followed by the summary itself."""
    
    # Encode and send to model
    encoding = tokenizer(input_string, return_tensors='pt', padding=True, truncation=True, max_length=10000)
    encoding = {k: v.to("cuda") for k, v in encoding.items()}

    # Generate output with attention mask and specified pad token id
    with torch.no_grad():
        output = model.generate(
            **encoding,
            max_new_tokens=1000,  # Adjust as necessary to get complete summaries
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id  # Use the pad_token_id explicitly
        )
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    print(decoded_output)

    # Extract the summary from the output
    summary_start = decoded_output.find('Summary: ')
    summary = decoded_output[summary_start+len('Summary: '):] if summary_start != -1 else "Summary not found."
    print(summary)
    return summary

# Load the CSV file
df = pd.read_csv('reports_with_few_shots.csv')

# Generate summaries for the reports in the dataframe
df['Llama3cot'] = df['REPORT'].progress_apply(summarize_medical_document)

# Save the updated DataFrame back to the same file
df.to_csv('reports_with_few_shots.csv', index=False)
