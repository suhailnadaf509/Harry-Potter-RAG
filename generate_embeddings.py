import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

def main():
    print("Generating embeddings for Harry Potter documents...")
    
    # Initialize the model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    documents_dir = os.path.join(current_dir, "documents")
    
    # Check if documents directory exists
    if not os.path.exists(documents_dir):
        print(f"Error: Documents directory not found at {documents_dir}")
        print("Please create a 'documents' folder with your text files.")
        return
    
    # Process text files
    documents = []
    for filename in os.listdir(documents_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(documents_dir, filename)
            try:
                print(f"Processing {filename}...")
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Split into chunks of ~1000 characters
                chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
                for chunk in chunks:
                    documents.append({
                        "content": chunk,
                        "source": filename
                    })
            except Exception as e:
                print(f"Error reading {filename}: {str(e)}")
    
    if not documents:
        print("No documents found. Please add .txt files to the 'documents' folder.")
        return
    
    print(f"Found {len(documents)} document chunks.")
    
    # Extract content for embeddings
    contents = [doc["content"] for doc in documents]
    
    # Generate embeddings
    print("Generating embeddings... (this may take a while)")
    embeddings = model.encode(contents)
    
    # Save documents and embeddings
    documents_path = os.path.join(current_dir, "documents.json")
    embeddings_path = os.path.join(current_dir, "embeddings.npy")
    
    with open(documents_path, 'w', encoding='utf-8') as f:
        json.dump([doc["content"] for doc in documents], f)
    
    np.save(embeddings_path, embeddings)
    
    print(f"Successfully saved {len(documents)} document chunks and embeddings.")
    print(f"Documents saved to: {documents_path}")
    print(f"Embeddings saved to: {embeddings_path}")

if __name__ == "__main__":
    main() 