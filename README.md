# Harry Potter Quiz App

A Streamlit application that uses RAG (Retrieval-Augmented Generation) to answer questions about the Harry Potter universe.

## Features

- Ask questions about Harry Potter
- Retrieves relevant passages from the books
- Uses Groq's LLM to generate accurate answers based on the retrieved content

## Local Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Groq API key:
   ```
   GROQ_API_KEY=your-api-key-here
   ```
4. Run the database formation script to create the vector database:
   ```
   python database_formation.py
   ```
5. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Deploying to Streamlit Cloud

1. Fork this repository to your GitHub account
2. Sign up for [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app and connect it to your GitHub repository
4. In the app settings, add your Groq API key as a secret:
   - Go to "Advanced settings" > "Secrets"
   - Add a new secret with the key `GROQ_API_KEY` and your API key as the value
5. Deploy the app

## Project Structure

- `app.py`: The main Streamlit application
- `database_formation.py`: Script to create the vector database from text files
- `documents/`: Directory containing Harry Potter text files
- `db/`: Directory where the vector database is stored
- `requirements.txt`: List of Python dependencies

## Notes

- The app requires SQLite 3.35.0 or higher, which is available on Streamlit Cloud
- Make sure to upload your Harry Potter text files to the `documents/` directory before running `database_formation.py`
