# Import SQLite fix first, before any other imports
import sqlite_fix

import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    books_dir = os.path.join(current_dir, "documents")
    db_dir = os.path.join(current_dir, "db")
    persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")
    
    logger.info(f"Books directory: {books_dir}")
    logger.info(f"Persistent directory: {persistent_directory}")
    
    # Create db directory if it doesn't exist
    os.makedirs(db_dir, exist_ok=True)
    
    # Check if the Chroma vector store already exists
    if not os.path.exists(persistent_directory):
        logger.info("Persistent directory does not exist. Initializing vector store...")
        
        # Ensure the books directory exists
        if not os.path.exists(books_dir):
            raise FileNotFoundError(
                f"The directory {books_dir} does not exist. Please check the path."
            )
        
        # List all text files in the directory
        book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]
        
        if not book_files:
            raise FileNotFoundError("No .txt files found in the documents directory.")
        
        # Read the text content from each file and store it with metadata
        documents = []
        for book_file in book_files:
            try:
                file_path = os.path.join(books_dir, book_file)
                logger.info(f"Loading {book_file}...")
                
                # Try different encodings if needed
                try:
                    loader = TextLoader(file_path, encoding='utf-8')
                    book_docs = loader.load()
                except UnicodeDecodeError:
                    logger.warning(f"UTF-8 encoding failed for {book_file}, trying with latin-1")
                    loader = TextLoader(file_path, encoding='latin-1')
                    book_docs = loader.load()
                
                for doc in book_docs:
                    # Add metadata to each document indicating its source
                    doc.metadata = {"source": book_file}
                    documents.append(doc)
                logger.info(f"Successfully loaded: {book_file}")
            except Exception as e:
                logger.error(f"Error loading {book_file}: {str(e)}")
                continue
        
        if not documents:
            raise ValueError("No documents were successfully loaded.")
        
        # Split the documents into chunks with overlap
        logger.info("Splitting documents into chunks...")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        
        # Display information about the split documents
        logger.info(f"Number of document chunks: {len(docs)}")
        
        # Create and persist the vector store
        logger.info("Creating and persisting vector store...")
        db = Chroma.from_documents(
            docs, embeddings, persist_directory=persistent_directory)
        logger.info("Finished creating and persisting vector store.")
        
        # Verify the database is loaded
        count = db._collection.count()
        logger.info(f"Vector store created with {count} documents.")
        
    else:
        logger.info("Loading existing vector store...")
        db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
        count = db._collection.count()
        logger.info(f"Vector store loaded with {count} documents.")
    
    logger.info("Database formation complete!")
    return db

if __name__ == "__main__":
    main()