import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import os
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import io
import numpy as np
import pandas as pd
import re

# CSS remains the same
css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/rdZC7LZ/Photo-logo-1.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

def extract_text_from_image(image):
    try:
        # Convert image to grayscale
        gray_image = image.convert('L')
        
        # Apply thresholding to binarize the image
        thresholded_image = gray_image.point(lambda x: 0 if x < 128 else 255, '1')
        
        # Use Tesseract with specific configuration
        custom_config = r'--oem 3 --psm 6'  # Use LSTM OCR Engine and assume a single uniform block of text
        text = pytesseract.image_to_string(thresholded_image, config=custom_config)
        return text
    except Exception as e:
        st.error(f"Error in OCR processing: {str(e)}")
        return ""

def get_pdf_text(pdf_docs):
    try:
        text = ""
        all_texts = []  # Store text from each PDF separately
        for pdf in pdf_docs:
            pdf_content = pdf.read()
            current_text = ""
            
            try:
                # First try normal PDF text extraction
                pdf_reader = PdfReader(io.BytesIO(pdf_content))
                page_texts = []
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text.strip():
                        page_texts.append(page_text)
                
                # If no text was extracted, try OCR
                if not page_texts:
                    st.info(f"No text found in {pdf.name} using direct extraction. Attempting OCR...")
                    images = convert_from_bytes(pdf_content)
                    for image in images:
                        # Rotate the image if necessary
                        rotated_image = correct_image_orientation(image)
                        page_text = extract_text_from_image(rotated_image)
                        if page_text.strip():
                            page_texts.append(page_text)
                
                current_text = "\n".join(page_texts)
                if current_text.strip():
                    all_texts.append({
                        "filename": pdf.name,
                        "content": current_text
                    })
                text += current_text + "\n\n"
                
            except Exception as e:
                st.warning(f"Error processing {pdf.name}: {str(e)}")
                continue

        if not text.strip():
            st.error("No text could be extracted from any of the PDFs")
            return None, None
            
        st.success("Successfully extracted text from PDFs!")
        return text, all_texts
        
    except Exception as e:
        st.error(f"Error processing PDFs: {str(e)}")
        return None, None

def correct_image_orientation(image):
    """Correct the orientation of the image if it is rotated."""
    # Use pytesseract to detect orientation
    osd = pytesseract.image_to_osd(image)
    rotation = int(re.search(r'Rotate:\s*(\d+)', osd).group(1))
    
    if rotation == 90:
        return image.rotate(-90, expand=True)
    elif rotation == 180:
        return image.rotate(180, expand=True)
    elif rotation == 270:
        return image.rotate(90, expand=True)
    
    return image  # No rotation needed

def get_text_chunks(text):
    if text is None:
        return None
    try:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        if not chunks:
            st.error("No text chunks were created")
            return None
        return chunks
    except Exception as e:
        st.error(f"Error splitting text: {str(e)}")
        return None

def get_vectorstore(text_chunks):
    if text_chunks is None:
        return None
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        
        # Store chunks for display
        st.session_state.text_chunks = text_chunks
        st.session_state.embeddings_model = embeddings
        
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def create_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.7,
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )

def get_conversation_chain(vectorstore):
    if vectorstore is None:
        return None
    try:
        llm = create_llm()
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        )
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            memory=memory,
            return_source_documents=True,
            chain_type="stuff"
        )
        return conversation_chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        return None

def handle_userinput(user_question):
    if not st.session_state.get("vector_store_created", False):
        st.error("Please upload and process PDFs first!", icon="🚨")
        return
    
    if st.session_state.conversation is None:
        st.error("Conversation chain not initialized properly", icon="🚨")
        return

    try:
        response = st.session_state.conversation.invoke({
            "question": user_question
        })
        
        # Display the answer and relevant chunks
        answer = response['answer']
        source_docs = response['source_documents']

        # Display answer
        st.write(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)
        
        # Display relevant chunks
        with st.expander("View Source Chunks"):
            for i, doc in enumerate(source_docs):
                st.markdown(f"**Relevant Chunk {i+1}:**")
                st.write(doc.page_content)
                st.markdown("---")
                
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")

def display_pdf_content():
    if "all_pdf_texts" in st.session_state and st.session_state.all_pdf_texts:
        for pdf_text in st.session_state.all_pdf_texts:
            with st.expander(f"Content of {pdf_text['filename']}"):
                st.text(pdf_text['content'])

def display_vector_data():
    if "text_chunks" in st.session_state and "embeddings_model" in st.session_state:
        chunks = st.session_state.text_chunks
        embeddings_model = st.session_state.embeddings_model
        
        # Display chunks
        with st.expander("Text Chunks"):
            for i, chunk in enumerate(chunks):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(chunk)
                st.markdown("---")
        
        # Calculate and display sample embeddings
        if st.button("Calculate Sample Embeddings"):
            with st.spinner("Calculating embeddings..."):
                # Take first chunk as sample
                sample_embedding = embeddings_model.embed_query(chunks[0])
                
                # Create DataFrame for visualization
                df = pd.DataFrame({
                    'Dimension': range(len(sample_embedding)),
                    'Value': sample_embedding
                })
                
                # Display embedding visualization
                st.line_chart(df.set_index('Dimension'))
                st.write("Sample embedding vector (first 10 dimensions):")
                st.write(sample_embedding[:10])

def main():
    st.set_page_config(page_title="Enhanced PDF Chat", page_icon="📚", layout="wide")
    st.write(css, unsafe_allow_html=True)

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "vector_store_created" not in st.session_state:
        st.session_state.vector_store_created = False

    st.header("Enhanced PDF Chat with Gemini 📚")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Chat", "PDF Content", "Vector Data"])
    
    with tab1:
        # Chat interface
        with st.sidebar:
            st.subheader("Document Upload")
            pdf_docs = st.file_uploader(
                "Upload your PDFs here and click 'Process'",
                accept_multiple_files=True,
                type=['pdf']
            )
            
            if st.button("Process"):
                if not pdf_docs:
                    st.error("Please upload at least one PDF file.")
                    return
                
                with st.spinner("Processing your PDFs..."):
                    # Reset the state
                    st.session_state.vector_store_created = False
                    st.session_state.conversation = None
                    
                    # Process PDFs
                    raw_text, all_texts = get_pdf_text(pdf_docs)
                    if raw_text and all_texts:
                        st.session_state.all_pdf_texts = all_texts  # Store for display
                        
                        text_chunks = get_text_chunks(raw_text)
                        if text_chunks:
                            vectorstore = get_vectorstore(text_chunks)
                            if vectorstore:
                                conversation_chain = get_conversation_chain(vectorstore)
                                if conversation_chain:
                                    st.session_state.conversation = conversation_chain
                                    st.session_state.vector_store_created = True
                                    st.success("Processing complete!")

        user_question = st.text_input(
            "Ask a question about your documents:",
            placeholder="What would you like to know about the PDFs?"
        )
        
        if user_question:
            handle_userinput(user_question)
    
    with tab2:
        # PDF Content display
        display_pdf_content()
    
    with tab3:
        # Vector data display
        display_vector_data()

if __name__ == '__main__':
    main()