from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import torch
import pandas as pd
from tqdm import tqdm
import faiss

# Setup tqdm for pandas progress bar
tqdm.pandas()

# Load the synthetic clinical notes dataset as knowledge base
print("Loading the knowledge base...")
knowledge_base = load_dataset("Technoculture/synthetic-clinical-notes-embedded")['train']

# Preprocess and embed the knowledge base
print("Embedding the knowledge base...")
retriever_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
retriever_model.eval()

knowledge_texts = knowledge_base['output']  # Use the 'output' field for retrieval
kb_embeddings = []

# Use tqdm for embedding progress
for text in tqdm(knowledge_texts, desc="Embedding knowledge base"):
    kb_embeddings.append(retriever_model.encode(text, convert_to_tensor=True).cpu().detach().numpy())
kb_embeddings = torch.tensor(kb_embeddings)

# Create FAISS index
print("Creating FAISS index...")
index = faiss.IndexFlatL2(kb_embeddings.shape[1])  # Assuming 384-dimensional embeddings
index.add(kb_embeddings.numpy())

# Load the LLaMA model and tokenizer
print("Loading the LLaMA model...")
llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="balanced")
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="balanced")

if llama_tokenizer.pad_token is None:
    llama_tokenizer.pad_token = llama_tokenizer.eos_token  # Ensure pad token is set for padding handling

llama_model.eval()

# Load the few-shot examples CSV
few_shot_csv_path = '/cronus_data/ksingh/run/final/few_shot.csv'
print("Loading few-shot examples...")
few_shot_df = pd.read_csv(few_shot_csv_path)

# Generate the few-shot examples for the prompt
few_shot_examples = ""
for _, row in few_shot_df.iterrows():
    few_shot_examples += f"Example: {row['ChatGPT 3.5']}\n\n"

# Function to retrieve context from the knowledge base
def retrieve_context(document, top_k=3):
    doc_embedding = retriever_model.encode(document, convert_to_tensor=True)
    distances, indices = index.search(doc_embedding.cpu().detach().numpy().reshape(1, -1), top_k)
    retrieved_docs = "\n".join([knowledge_texts[i] for i in indices[0]])
    return retrieved_docs

# Function to summarize medical documents with RAG
def summarize_medical_document_with_rag(document):
    try:
        # Retrieve relevant context
        retrieved_context = retrieve_context(document)
        
        # Construct input string with retrieved context
        input_string = f"""
        You are a helpful assistant trained to process medical documentation for legal review.
        Few-Shot Examples: {few_shot_examples} 
        Retrieved Context: {retrieved_context}
        Here is a medical document: {document}
        Analyze the document and provide a step-by-step explanation of key findings, treatments, and any notable outcomes. 
        Ensure your response includes a section starting with "Summary:" followed by a concise summary for legal professionals."""

        # Debug: Print input string
        print("Input to model:")
        print(input_string)

        # Encode and send to LLaMA 3
        encoding = llama_tokenizer(input_string, return_tensors='pt', padding=True, truncation=True, max_length=5000)
        encoding = {k: v.to("cuda") for k, v in encoding.items()}

        # Generate output
        with torch.no_grad():
            output = llama_model.generate(
                **encoding,
                max_new_tokens=1500,  # Adjust as necessary
                num_return_sequences=1,
                pad_token_id=llama_tokenizer.pad_token_id,
                temperature=0.7,
                top_p=0.9
            )
        decoded_output = llama_tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract summary from output
        summary_start = decoded_output.find('Summary:')
        if summary_start != -1:
            summary = decoded_output[summary_start + len('Summary: '):].strip()
        else:
            summary = "Summary not found in the output. Please check the model's response."

        print("Generated Summary:")
        print(summary)
        return decoded_output
    except Exception as e:
        print(f"Error generating summary for document: {e}")
        return "Error generating summary."

# Load the medical reports
print("Loading medical reports...")
df = pd.read_csv('/cronus_data/ksingh/run/final/reports.csv')

# Generate summaries for the reports
print("Generating summaries...")
df['Llama3rag'] = df['REPORT'][:7].progress_apply(summarize_medical_document_with_rag)

# Save the updated DataFrame
print("Saving results...")
df.to_csv('/cronus_data/ksingh/run/final/reports.csv', index=False)
