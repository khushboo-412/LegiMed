# LegiMed: Medical Document Summarizer for Legal Professionals

LegiMed is an advanced AI-driven tool designed to streamline the process of summarizing medical documents for legal professionals. By leveraging state-of-the-art language models, LegiMed provides concise, accurate, and legally pertinent summaries of complex medical records, enhancing efficiency and clarity in legal cases involving medical data.

## Features

- **Multiple Language Model Support**: Integrates several cutting-edge models:
  - **T5 and FLAN-T5**: For robust, context-aware text generation.
  - **BART and BioBART**: Specialized in reconstructing narratives from medical texts.
  - **ChatGPT-3.5 and ChatGPT-4**: For interactive, conversational user queries.
  - **Llama3 and Llama3.2**: Utilized for advanced inference strategies like zero-shot and few-shot learning, as well as Chain-of-Thought reasoning.
- **Zero-Shot and Few-Shot Learning**: Using Llama3 for generating summaries without extensive training on domain-specific data.
- **Chain-of-Thought Prompting**: Enhances the ability of models to handle complex queries by guiding them through intermediate steps.
- **Medical Embeddings**: Incorporates medical-specific embeddings to improve accuracy in interpreting medical jargon.
- **Retrieval-Augmented Generation (RAG)**: Combines the power of retrieval with generative capabilities to enhance factuality and detail in summaries.

## Installation

Clone the repository to your local machine:
```bash
git clone https://github.com/your-username/LegiMed.git
cd LegiMed
```

## Usage

To start summarizing medical documents, follow these steps:
1. Prepare your medical documents in a text format.
2. Run the summarizer script with the desired model:
3. Review the generated summary from the output file.

## Contributing

We welcome contributions from the community. If you'd like to contribute, please fork the repository and use a pull request for your contributions.

## Acknowledgements

We acknowledge the developers and researchers behind the T5, FLAN T5, BART, BioBART, ChatGPT, and Llama models for providing the tools that power LegiMed.
