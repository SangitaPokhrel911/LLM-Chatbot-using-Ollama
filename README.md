## Project Overview
This project involves building a RAG (Retrieval-Augmented Generation) based LLM chatbot using Ollama, Streamlit, and ChromaDB.
Users can directly ask questions and receive offline answers retrieved from the uploaded documents (supports multiple document uploads).

**Setup Instructions**
1. Create a Virtual Environment
Open your terminal inside the project folder and run:

        # For Windows
        python -m venv venv

        # For macOS / Linux
        python3 -m venv venv


2. Activate the Virtual Environment

**For Windows**
venv\Scripts\activate

**For macOS / Linux**
source venv/bin/activate
You should see (venv) at the start of your terminal prompt after activation.

3. Install Dependencies
Once the virtual environment is active, install the required packages:

pip install -r requirements.txt

4. Run the Streamlit Application
In the same terminal (with the environment activated), run:

streamlit run app.py

This will start the Streamlit server, and you can access the chatbot in your browser:
http://localhost:8501

5. Ask questions to your chatbot after uploading the documents. 
