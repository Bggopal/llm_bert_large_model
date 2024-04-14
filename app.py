import os
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import streamlit as st

# Define the path to the directory containing the model files
MODEL_DIR = "dir"

# Load the question answering model from the local directory
@st.cache_data()
def load_qa_model():
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, "tokenizer"))
    model = AutoModelForQuestionAnswering.from_pretrained(os.path.join(MODEL_DIR, "model"))
    return {'tokenizer': tokenizer, 'model': model}

# Load the model
qa_model = load_qa_model()

# Streamlit app
st.title("Document-based Question Answering System")

uploaded_file = st.file_uploader("Upload your document:", type=['txt', 'pdf'])

if uploaded_file is not None:
    st.write("Document Uploaded Successfully!")
    document_text = uploaded_file.read().decode("utf-8")

    st.write("Document Text:")
    st.write(document_text)

    question = st.text_input("Ask a question based on the document:")

    if st.button("Get Answer"):
        # Initialize the pipeline for question answering
        nlp = pipeline("question-answering", model=qa_model['model'], tokenizer=qa_model['tokenizer'])

        # Get the answer
        answer = nlp(question=question, context=document_text)

        st.write("Answer:", answer['answer'])
        st.write("Confidence:", answer['score'])
