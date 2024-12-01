from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
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

# Load the LLaMA 3 model and tokenizer
llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="balanced")
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="balanced")

if llama_tokenizer.pad_token is None:
    llama_tokenizer.pad_token = llama_tokenizer.eos_token  # Ensure pad token is set for padding handling

llama_model.eval()

# Load PubMedBERT for medical embeddings
bert_model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
bert_tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

bert_model.eval()
bert_model.to("cuda")

# Function to extract embeddings using PubMedBERT
def get_bert_embeddings(document):
    inputs = bert_tokenizer(document, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
    # Use the mean of the last hidden state as the embedding
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# Function to summarize medical documents with Chain-of-Thought prompting
def summarize_medical_document_with_embeddings(document):
    # Get the embeddings for the document
    bert_embeddings = get_bert_embeddings(document)

    # Integrate embeddings into the input for LLaMA 3
    input_string = f"""
    You are a helpful assistant trained to process medical documentation for legal review.
    Few-Shot Examples: {few_shot_examples} 
    Here is a medical document: {document}
    Analyze the document and provide a step-by-step explanation of key findings, treatments, and any notable outcomes. 
    Ensure your response includes a section starting with "Summary:" followed by a concise summary for legal professionals."""
    
    # Debug: Print input to the model
    print("Input to the model:")
    print(input_string)

    # Encode and send to LLaMA 3
    encoding = llama_tokenizer(input_string, return_tensors='pt', padding=True, truncation=True, max_length=4000)
    encoding = {k: v.to("cuda") for k, v in encoding.items()}

    # Generate output with LLaMA 3
    with torch.no_grad():
        output = llama_model.generate(
            **encoding,
            max_new_tokens=1000,  
            num_return_sequences=1,
            pad_token_id=llama_tokenizer.pad_token_id  
        )
    decoded_output = llama_tokenizer.decode(output[0], skip_special_tokens=True)

    # Debug: Print model output
    print("Model output:")
    print(decoded_output)

    # Extract the summary from the output
    summary_start = decoded_output.find('Summary:')
    if summary_start != -1:
        summary = decoded_output[summary_start + len('Summary: '):].strip()
    else:
        print("Model did not return a summary. Full output:")
        print(decoded_output)
        summary = "Summary not found. Please check the input and model behavior."
    print(summary)
    return summary

# Load the CSV file
df = pd.read_csv('reports_with_few_shots.csv')

# Generate summaries for the reports in the dataframe
df['Llama3emb'] = df['REPORT'].progress_apply(summarize_medical_document_with_embeddings)

# Save the updated DataFrame back to the same file
df.to_csv('reports_with_few_shots.csv', index=False)
