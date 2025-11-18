ai-rental-analysis/
├── app.py                    # Main Streamlit application
├── legal_analyzer.py         # LangGraph workflow and analysis logic
├── contract_analysis_rag.py  # RAG system with Legal-BERT embeddings
├── chunking_system.py        # Document chunking with overlap
├── mychunk.py                # Chunk data model
├── __init__.py               # Package initialization
├── uploads/                  # Uploaded files (created automatically)
├── .env                      # Environment variables
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── tests/                    # Test files (optional)

**Create virtual Environment**
python -m venv venv

# Windows (PowerShell) - Fix execution policy first:
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1

# Windows (cmd)
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

**Install Dependencies**
pip install streamlit pandas python-docx pdfplumber pillow pytesseract
pip install sentence-transformers transformers torch
pip install langgraph langchain-core numpy scikit-learn
pip install python-dotenv  # Optional, for .env file support

**Configure tessaract**
# Path to Tesseract executable (update for your system)
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
**# Windows default path (update if different)
**# macOS/Linux: usually auto-detected or /usr/local/bin/tesseract

**# Optional: AWS S3 credentials for cloud storage
**# AWS_ACCESS_KEY_ID=your_key
**# AWS_SECRET_ACCESS_KEY=your_secret
**# AWS_REGION=us-east-1

**Start the Streanlit Server**
streamlit run app.py
## Updates in Branch rohi
- Risk meter graph implemented
- Heavy ML model replaced with lightweight model
- Added clause confidence scoring
- UI improvements for better usability
- Model now loads only once for performance optimization
