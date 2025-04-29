
# 📚 Persistent Multi-Modal Q&A & Data Analysis Chatbot

A Streamlit-based application enabling conversational Q&A over uploaded documents (PDF, DOCX, TXT, PNG, JPG, CSV, XLSX, JSON) using Google Gemini models. Supports persistent chat sessions, natural language querying of data files via NL-to-SQL, and multi-modal context retrieval via ChromaDB.

---

## WEBSITE LINK: 

https://mando-chatbot.streamlit.app/

## ✨ Features

- **Multi-Modal Input**  
  Upload and process a wide range of file types including:  
  `PDF`, `DOCX`, `TXT`, `PNG`, `JPG/JPEG`, `CSV`, `XLSX`, `JSON`.

- **Text & Image Q&A (RAG)**  
  Ask questions about document or image contents using Retrieval-Augmented Generation powered by ChromaDB.

- **Natural Language Data Analysis**  
  Upload structured data files and ask questions like:  
  > "What is the total sales?" or "Show average price per category."  
  Queries are converted to SQL and executed on chat-specific SQLite databases.

- **Persistent Multi-Chat Sessions**  
  Each chat session maintains its own set of files, data context, and message history.

- **Customizable Model Selection**  
  Choose from available Gemini models via a sidebar dropdown.

- **URL Content Extraction**  
  Automatically crawl and process content from URLs found in uploaded documents.

- **Modular Codebase**  
  Clean architecture for easy maintenance and extension.

---

## 🗂️ Project Structure

```
multimodal_qa_data_app/
├── main_app.py                 # Streamlit UI & session management
├── config.py                   # Configs for paths, model selection, DBs
├── utils.py                    # Common utility functions
├── file_parsers.py             # Document/data file parsing & triggers
├── image_processor.py          # OCR or vision model interface
├── web_crawler.py              # Web crawling & URL content extraction
├── vector_store.py             # ChromaDB interface & RAG context handling
├── qa_engine.py                # Main Q&A handler (RAG + data logic)
├── DataImporter_Gemini.py      # Data loader for chat-specific SQLite DBs
├── VannaReplica_Gemini.py      # NL-to-SQL conversion logic
├── VannaDataAnalyzer_Gemini.py # SQL engine & natural language explanation
├── run_vanna_Gemini.py         # CLI for testing Vanna functionality
├── chat_databases/             # Folder for session-specific SQLite files
└── .streamlit/
    └── secrets.toml            # (Optional) Local API key config
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repo
```bash
git clone https://github.com/senthilnathan01/chatbot_for_enterprise
cd multimodal_qa_data_app
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
> 🔧 Note: `PyMuPDF` may require system-specific dependencies. Refer to [PyMuPDF installation guide](https://pymupdf.readthedocs.io/) if needed.

### 4. Set API Key for Gemini
- **Option A**: Enter directly in the sidebar input (session-only)
- **Option B**: Environment variable  
  ```bash
  export GOOGLE_API_KEY='your-key'       # macOS/Linux  
  set GOOGLE_API_KEY=your-key            # Windows CMD  
  $env:GOOGLE_API_KEY="your-key"         # PowerShell
  ```
- **Option C**: `.streamlit/secrets.toml`  
  ```toml
  GOOGLE_API_KEY = "your-key"
  ```

---

## 🚀 Running the App

```bash
streamlit run main_app.py
```

```bash
python -m streamlit run main_app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📊 How Data Analysis Works

1. **Data Import:**  
   Upload CSV, XLSX, or JSON files.  
   `file_parsers.py` uses `DataImporter_Gemini.py` to:
   - Create a chat-specific SQLite DB inside `chat_databases/`
   - Clean column names and detect types
   - Load the file content into SQL tables
   - Add a vector marker for context-awareness in ChromaDB

2. **Query Execution:**  
   `qa_engine.py` checks if the question is data-related.  
   If so:
   - Uses `VannaDataAnalyzer_Gemini.py` to extract schema
   - Sends schema + question to Gemini via `VannaReplica_Gemini.py`
   - Generates and runs SQL
   - Returns results + SQL explanation in natural language

3. **Chat Isolation:**  
   Each session has a unique DB file. Deleting a chat also deletes its database.

---

## 📝 Notes

- Large files may take time to process or query.
- Review auto-generated SQL queries for complex tasks.
- Imported data is **session-specific** and tied to Streamlit runtime state.
- Deleting or restarting the app will clear imported data unless persisted manually.

---

## ✅ Final Setup Checklist

- [ ] Ensure `.streamlit/secrets.toml` exists (if used)
- [ ] Create `chat_databases/` folder or let it auto-create on first run
- [ ] Install requirements: `pip install -r requirements.txt`
- [ ] Run the app: `streamlit run main_app.py` or `python -m streamlit run main_app.py`

---

## 📬 Contributions Welcome!

Found a bug, want to add a feature, or improve performance?  
Feel free to open an issue or submit a PR!

