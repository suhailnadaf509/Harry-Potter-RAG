import os
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set up page
st.set_page_config(page_title="🧙🧙‍♂️🪄🔗 Harry Potter Quiz")
st.title('🧙‍♂️🪄 Harry Potter Quiz')

# Initialize Groq client if API key is available
groq_api_key = os.getenv("GROQ_API_KEY")
client = None

if groq_api_key:
    try:
        from groq import Groq
        client = Groq(api_key=groq_api_key)
    except Exception as e:
        st.error(f"Error initializing Groq client: {str(e)}")
        st.info("Continuing without LLM capabilities. Some features may be limited.")
else:
    st.warning("GROQ_API_KEY not found in environment variables. LLM features will be disabled.")

with st.form('my_form'):
    question = st.text_input("Ask a Harry Potter question here")
    submitted = st.form_submit_button("Submit")

if submitted:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    persistent_directory = os.path.join(
        current_dir, "db", "chroma_db_with_metadata")

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
        else:
            st.info("LLM response not available. Please check your GROQ_API_KEY.")
    except Exception as e:
        st.error(f"Error processing request: {str(e)}")
        st.info("Please make sure the vector database has been properly initialized.")