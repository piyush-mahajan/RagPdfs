# Enhanced PDF Chat with Gemini 📚

This Streamlit-based application allows you to upload PDFs, extract their text (using both standard extraction and OCR), store them in a FAISS vector store, and interact with the content using Google's Gemini AI model.

---

## Features

- **PDF Text Extraction**: Extracts text from PDFs using `PyPDF2`.
- **OCR Processing**: Uses Tesseract OCR to extract text from image-based PDFs.
- **Text Chunking**: Splits extracted text into manageable chunks using LangChain's `CharacterTextSplitter`.
- **Vector Store**: Converts text into embeddings using HuggingFace models and stores them in a FAISS vector store.
- **Conversational AI**: Interact with your PDF content through a chat interface powered by Google's Gemini AI.
- **Interactive UI**: Built with Streamlit, featuring a chat interface and expandable views for extracted text and vector data.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-folder>


Here's the README content in code format:

markdown
Copy
Edit
# Enhanced PDF Chat with Gemini 📚

This Streamlit-based application allows you to upload PDFs, extract their text (using both standard extraction and OCR), store them in a FAISS vector store, and interact with the content using Google's Gemini AI model.

---

## Features

- **PDF Text Extraction**: Extracts text from PDFs using `PyPDF2`.
- **OCR Processing**: Uses Tesseract OCR to extract text from image-based PDFs.
- **Text Chunking**: Splits extracted text into manageable chunks using LangChain's `CharacterTextSplitter`.
- **Vector Store**: Converts text into embeddings using HuggingFace models and stores them in a FAISS vector store.
- **Conversational AI**: Interact with your PDF content through a chat interface powered by Google's Gemini AI.
- **Interactive UI**: Built with Streamlit, featuring a chat interface and expandable views for extracted text and vector data.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-folder>
2. Set Up a Virtual Environment (Optional but Recommended)
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
If a requirements.txt file is not provided, you can manually install the required packages:

bash
pip install streamlit PyPDF2 langchain langchain_huggingface langchain_google_genai langchain_community google-generativeai pytesseract pdf2image pillow numpy pandas


4. Install Additional System Dependencies
Tesseract OCR: Required for extracting text from images.

Ubuntu/Debian: sudo apt-get install tesseract-ocr
MacOS: brew install tesseract
Windows: Download the installer
Poppler: Required for pdf2image to convert PDFs to images.

Ubuntu/Debian: sudo apt-get install poppler-utils
MacOS: brew install poppler
Windows: Download Poppler for Windows

5. Set Up Environment Variables
Create a .streamlit/secrets.toml file and add your Google API key:


[GOOGLE_API_KEY]
GOOGLE_API_KEY = "your_google_api_key_here"


Running the Application
Start the Streamlit app with:

bash
Copy
Edit
streamlit run <your_script_name>.py
Replace <your_script_name>.py with the actual file name (e.g., app.py).

Usage
Upload PDFs: Use the sidebar to upload multiple PDF files.
Process PDFs: Click the "Process" button to extract text and create embeddings.
Ask Questions: Use the chat interface to ask questions about the content in your PDFs.
View Data: Explore extracted text and vector embeddings using the tabs provided.