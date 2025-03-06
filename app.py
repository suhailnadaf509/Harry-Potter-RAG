import os
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set up page
st.set_page_config(page_title="🧙🧙‍♂️🪄🔗 Harry Potter Quiz")
st.title('🧙‍♂️🪄 Harry Potter Quiz')

# Initialize embeddings
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
except Exception as e:
    st.error(f"Error initializing embeddings: {str(e)}")
    embeddings = None

# Initialize Groq client if API key is available
groq_api_key = os.getenv("GROQ_API_KEY")
client = None

if groq_api_key:
    try:
        # Try different import patterns for different groq package versions
        try:
            from groq import Groq
            client = Groq(api_key=groq_api_key)
        except (ImportError, AttributeError):
            try:
                import groq
                client = groq.Client(api_key=groq_api_key)
            except (ImportError, AttributeError):
                st.error("Could not import Groq client. Please check the groq package version.")
    except Exception as e:
        st.error(f"Error initializing Groq client: {str(e)}")
        st.info("Continuing without LLM capabilities. Some features may be limited.")
else:
    st.warning("GROQ_API_KEY not found in environment variables. LLM features will be disabled.")

with st.form('my_form'):
    question = st.text_input("Ask a Harry Potter question here")
    submitted = st.form_submit_button("Submit")

if submitted:
    if not question:
        st.warning("Please enter a question.")
    else:
        # Try to load the vector database
        try:
            # Import here to avoid immediate SQLite errors
            from langchain_community.vectorstores import Chroma
            from langchain_core.messages import HumanMessage, SystemMessage
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            persistent_directory = os.path.join(
                current_dir, "db", "chroma_db_with_metadata")
            
            # Check if the directory exists
            if not os.path.exists(persistent_directory):
                st.error("Vector database not found. Please run the database_formation.py script first.")
                st.info("The database needs to be created before you can ask questions.")
            else:
                try:
                    db = Chroma(persist_directory=persistent_directory,
                                embedding_function=embeddings)
                    
                    # Retrieve relevant documents based on the query
                    retriever = db.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 3},
                    )
                    relevant_docs = retriever.invoke(question)
                    
                    # Display the relevant results with metadata
                    st.write("\n--- Relevant Documents ---")
                    for i, doc in enumerate(relevant_docs, 1):
                        st.write(f"Document {i}:\n{doc.page_content}\n")
                    
                    if client:
                        combined_input = (
                            "Here are some documents that might help answer the question: "
                            + question
                            + "\n\nRelevant Documents:\n"
                            + "\n\n".join([doc.page_content for doc in relevant_docs])
                            + "\n\nPlease provide a rough answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
                        )
                        
                        try:
                            # Try different API patterns
                            try:
                                completion = client.chat.completions.create(
                                    model="llama-3.3-70b-versatile",
                                    messages=[
                                        {
                                            "role": "system",
                                            "content": "You are a Harry Potter expert who answers questions about the Harry Potter universe."
                                        },
                                        {
                                            "role": "user",
                                            "content": combined_input
                                        }
                                    ]
                                )
                                st.write(completion.choices[0].message.content)
                            except AttributeError:
                                # Fallback for older API
                                completion = client.completions.create(
                                    model="llama-3.3-70b-versatile",
                                    prompt=f"You are a Harry Potter expert. {combined_input}",
                                    max_tokens=1000
                                )
                                st.write(completion.choices[0].text)
                        except Exception as e:
                            st.error(f"Error generating response: {str(e)}")
                            st.info("Could not generate LLM response. Showing only retrieved documents.")
                    else:
                        st.info("LLM response not available. Please check your GROQ_API_KEY.")
                except Exception as e:
                    st.error(f"Error accessing vector database: {str(e)}")
                    st.info("This may be due to SQLite version incompatibility. The deployment environment needs SQLite 3.35.0 or higher.")
                    st.markdown("""
                    ### Troubleshooting
                    
                    The error is likely due to SQLite version incompatibility. Chroma requires SQLite 3.35.0 or higher.
                    
                    Options to resolve this:
                    1. Deploy to a platform with a newer SQLite version
                    2. Use a different vector database that's compatible with the current environment
                    3. Consider using a managed service like Pinecone or Weaviate instead of local Chroma
                    
                    For more information, visit: https://docs.trychroma.com/troubleshooting#sqlite
                    """)
        except ImportError as e:
            st.error(f"Error importing required modules: {str(e)}")
            st.info("Please make sure all required packages are installed.")