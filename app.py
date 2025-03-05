import os
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="🧙🧙‍♂️🪄🔗 Harry Potter Quiz")
st.title('🧙‍♂️🪄 Harry Potter Quiz')

with st.form('my_form'):
    question = st.text_input("Ask a Harry Potter question here")
    submitted = st.form_submit_button("Submit")

if submitted:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    persistent_directory = os.path.join(
        current_dir, "db", "chroma_db_with_metadata")

    db = Chroma(persist_directory=persistent_directory,
                embedding_function=embeddings)

    # Retrieve relevant documents based on the query
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    relevant_docs = retriever.invoke(question)

    # Display the relevant results with metadata


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
    st.write("\n--- Relevant Documents ---")
    for i, doc in enumerate(relevant_docs, 1):
        st.write(f"Document {i}:\n{doc.page_content}\n")