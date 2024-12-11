import streamlit as st
import pandas as pd
import time
from difflib import SequenceMatcher
import os

# Enhanced CSS for styling
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #e3f2fd, #ffffff);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .title {
        color: #2C3E50;
        font-size: 50px;
        text-align: center;
        margin-bottom: 10px;
        font-weight: bold;
        letter-spacing: 1px;
    }
    .tagline {
        color: #16A085;
        font-size: 22px;
        text-align: center;
        margin-bottom: 30px;
        font-style: italic;
        font-weight: bold;
    }
    .description {
        color: #34495E;
        font-size: 18px;
        text-align: center;
        margin-bottom: 30px;
    }
    .input-text, .summary-box {
        font-size: 16px;
        color: #2C3E50; /* Dark blue text */
        background-color: #f1f8e9; /* Light green background */
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #16A085; /* Accent color border */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        white-space: pre-wrap;
        width: 80%;
        margin: auto;
        overflow: hidden;
    }
    .summary-box {
        background-color: #e3f2fd; /* Light blue background */
        border: 2px solid #1e88e5; /* Blue accent color border */
    }
    </style>
    """, unsafe_allow_html=True)

# Load the Excel file
@st.cache_data
def load_data():
    # Get the directory of the current script
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "reports.xlsx")
    
    # Check if the file exists
    if not os.path.exists(file_path):
        st.error(f"File 'reports.xlsx' not found in the directory: {current_dir}")
        st.stop()  # Stop the script execution
        
    # Load the Excel file
    return pd.read_excel(file_path, sheet_name='test')

# Load the data
sheet_data = load_data()

# Sidebar for user inputs
with st.sidebar:
    col1, col2 = st.columns([1, 4])
    with col1:
        current_dir = os.path.dirname(__file__)
        logo_path = os.path.join(current_dir, "logo.jpeg")
        if os.path.exists(logo_path):
            st.image(logo_path, width=50)
        else:
            st.text("Logo not found")
    with col2:
        st.markdown('<div class="sidebar-title">Medical Document Summarizer</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-subtext">Upload a text file and select a model to generate a summary.</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload a .txt file", type="txt")
    
    model_mapping = {
        "ChatGPT 3.5": "ChatGPT_3.5",
        "ChatGPT 4": "ChatGPT4",
        "Llama3 (using Interface)": "Llama3_Interface",
        "Llama3 (Zero Shot)": "Llama3_Zero_Shot",
        "LLama3 (Few Shots)": "LLama3_Few_Shots",
        "LLama3 (Chain of Thought)": "LLama3_Chain_of_Thought",
        "Llama3 (using BiomedNLP-PubMedBERT embeddings)" : "Llama3_embeddings",
        "Llama3 (using RAG)":"Llama3_RAG",
        "T5 (Zero Shot)": "T5_Zero_Shot",
        "T5 (Few Shots)" : "T5_Few Shots",
        "FlanT5 (Zero Shot)": "FlanT5_Zero",
        "FLANT5 (Few Shots)": "FLANT5_Few",
        "o1": "o1",
        "Deepseek": "deepseek"

    }
    model_choice = st.selectbox("Select a Model", list(model_mapping.keys()))
    st.markdown("---")
    st.write("Developed by Group 10")

# Main content
st.markdown('<div class="title">📄 LegiMed AI</div>', unsafe_allow_html=True)
st.markdown('<div class="tagline">Cut Through The Jargons, Focus On Justice.</div>', unsafe_allow_html=True)
st.markdown('<div class="description">Upload a text file, select a model, and generate a summary.</div>', unsafe_allow_html=True)

# Function to find the closest match
def find_closest_match(input_text, report_column):
    """
    Finds the closest match to input_text in the report_column using similarity ratios.
    """
    input_text = input_text.strip().lower()
    report_column = report_column.str.strip().str.lower()
    similarities = report_column.apply(lambda x: SequenceMatcher(None, input_text, x).ratio())
    max_similarity = similarities.max()
    if max_similarity > 0.8:  # Adjust threshold as needed
        return similarities.idxmax()  # Return index of the closest match
    return None

if uploaded_file is not None:
    # Read input text
    input_text = uploaded_file.read().decode("utf-8").strip()
    
    # Always show the Input Text section
    st.subheader("Input Text")
    st.markdown(
        f'<div class="input-text">{input_text}</div>',
        unsafe_allow_html=True
    )

    # Generate summary button
    if st.button("🚀 Generate Summary", help="Click to summarize the uploaded text!"):
        with st.spinner("Generating summary..."):
            # Find the closest match in the "REPORT" column
            match_index = find_closest_match(input_text, sheet_data["REPORT"])
            
            if match_index is not None:
                matching_row = sheet_data.iloc[match_index]
                selected_column = model_mapping[model_choice]
                
                try:
                    # Fetch the summary for the matching row and selected model
                    summary_output = matching_row[selected_column]

                    # Display the summary dynamically
                    summary_placeholder = st.empty()
                    lines = summary_output.splitlines()
                    displayed_summary = ""

                    for line in lines:
                        displayed_summary += line + "\n"
                        summary_placeholder.markdown(
                            f'<div class="summary-box">{displayed_summary.strip()}</div>',
                            unsafe_allow_html=True
                        )
                        time.sleep(0.5)

                    # Display the final formatted summary
                    summary_placeholder.markdown(
                        f'<div class="summary-box">{summary_output.strip()}</div>',
                        unsafe_allow_html=True
                    )
                except KeyError:
                    st.error(f"❌ Model '{model_choice}' not found in the dataset!")
                except Exception as e:
                    st.error(f"❌ An error occurred: {e}")
            else:
                st.error("❌ No matching report found in the dataset for the uploaded file!")

# Footer
st.markdown("""
    <hr>
    <div class="footer">
        &copy; 2024 LegiMed AI. All rights reserved.
    </div>
    """, unsafe_allow_html=True)
