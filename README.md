# Retrieval-Augmented Generation (RAG) System

This project implements a Retrieval-Augmented Generation (RAG) system using FastAPI, FAISS, and GPT-Neo. The goal is to create a backend service capable of answering user questions by retrieving relevant information from a set of documents and generating responses using a language model.

## Table of Contents
- [Project Overview](#project-overview)
- [File Structure](#file-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [API Endpoint](#api-endpoint)
- [Testing](#testing)
- [Logging](#logging)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)
- [Acknowledgments](#acknowledgments)
- [Notes](#notes)

---

## Project Overview

This system uses:
- **FAISS** for efficient document retrieval,
- **Sentence Transformers** for generating text embeddings,
- **GPT-Neo** as the language model for generating responses.

### Workflow
1. **Document Retrieval**: Uses FAISS to find the most relevant document chunks based on the user’s question.
2. **Response Generation**: Passes the retrieved context along with the question to GPT-Neo to generate a coherent, contextually accurate response.


## Setup Instructions
1. Clone the Repository
bash
Copy code
git clone https://github.com/shuvo881/RAG-System
cd RAG-System
2. Set Up the Environment
Create a virtual environment and install dependencies.

python -m venv venv
source venv/bin/activate   
On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
3. Configure Environment Variables
Create a .env file in the root directory with the following variables:


# Environment variables for API keys or other configurations
OPENAI_API_KEY=<your_openai_key_if_needed>
4. Preprocess Documents and Build FAISS Index
Place your document files in app/data/documents/. Run the following command to preprocess the documents and create the FAISS index:


python -m scripts.setup_index
Usage
To start the FastAPI application, use:


uvicorn app.main:app --reload
Navigate to http://127.0.0.1:8000/docs to access the interactive Swagger API documentation.

API Endpoint
POST /answer
Accepts a JSON payload with a question.
Retrieves relevant information from indexed documents and generates an answer.
Request Format

{
  "question": "What is Retrieval-Augmented Generation?"
}
Response Format

{
  "answer": "Generated answer based on the retrieved context."
}
Example Request
You can test the endpoint using curl:


curl -X 'POST' \
  'http://127.0.0.1:8000/answer' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"question": "What are common treatments for hypertension?"}'
# Testing
To run unit tests:

bash scripts/run_tests.sh
Tests
test_main.py: Tests the FastAPI endpoint /answer.
test_retrieval.py: Tests the retrieval process for accuracy and efficiency.
test_generation.py: Ensures that the generation model returns coherent answers.
test_pipeline.py: Tests the entire RAG pipeline.
Logging
Logs are configured to display informational messages, warnings, and errors. Key events, such as question reception, document retrieval, and answer generation, are logged. Errors include full stack traces for easier debugging.

Troubleshooting
Issue: ValueError: No columns to parse from file:

Ensure the document files are in a valid CSV format. If any files have inconsistent formatting, clean or reformat them.
Issue: TypeError: string indices must be integers, not 'str':

This error typically occurs in retrieve_documents when the retrieved documents are not in the expected format. Make sure retrieve_documents accesses data correctly, i.e., using doc['page_content'].
Issue: UserWarning: do_sample=False but temperature/top_p is set:

Set do_sample=True to enable temperature/top-p adjustments, or remove temperature and top_p to avoid the warning.
Future Enhancements
Model Upgrades: Experiment with larger or more advanced models (e.g., GPT-3 or fine-tuned models) for enhanced answer quality.
Extended Data Preprocessing: Add more sophisticated text processing, like entity extraction, to improve retrieval relevance.
Caching: Implement caching of frequently asked questions to improve response times for common queries.
Acknowledgments
This project leverages:

#LangChain for vector storage integration,
Hugging Face Transformers for language model support,
FAISS for high-speed vector similarity search.
Notes
Model Selection: The system uses GPT-Neo, but you can replace it with other open-source models.
Security: Only use trusted sources for serialized files, as FAISS deserialization requires allow_dangerous_deserialization=True.
Data Quality: Ensure that your documents are well-formatted to avoid issues during preprocessing and indexing.


## File Structure

```plaintext
RAG-System/
├── app/
│   ├── main.py               # FastAPI application and endpoint definitions
│   ├── config.py             # Configuration for environment variables and API keys
│   ├── services/
│   │   ├── retrieval.py      # Retrieval functions using FAISS
│   │   ├── generation.py     # Generation functions using GPT-Neo
│   │   ├── pipeline.py       # RAG pipeline combining retrieval and generation
│   │   └── logging.py        # Custom logging setup
│   ├── models/
│   │   ├── faiss_index/      # Folder for saved FAISS index files
│   │   └── embeddings/       # Folder for storing pre-trained embeddings if needed
│   ├── data/
│   │   └── documents/        # Raw document files or processed data for indexing
│   ├── tests/
│   │   ├── test_main.py      # Tests for the FastAPI endpoints
│   │   ├── test_retrieval.py # Unit tests for retrieval functions
│   │   ├── test_generation.py# Unit tests for generation functions
│   │   └── test_pipeline.py  # Tests for the RAG pipeline
│   └── utils/
│       ├── preprocess.py     # Helper functions for text cleaning and chunking
│       └── caching.py        # Functions for caching responses to common questions
├── requirements.txt          # List of dependencies for the project
├── README.md                 # Project documentation
├── .env                      # Environment variables for API keys and configurations
└── scripts/
    ├── setup_index.py        # Script to build and save the FAISS index
    └── run_tests.sh          # Script to run all tests and generate a report```



