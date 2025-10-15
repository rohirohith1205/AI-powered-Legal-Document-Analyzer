import streamlit as st
import os
from typing import Dict, Any, Optional
import pandas as pd
import docx
import pdfplumber
from PIL import Image
import pytesseract
import logging
from legal_analyzer import analyze_legal_document, GraphState

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configure Tesseract path (update as needed for your system)
tesseract_cmd = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
if os.path.exists(tesseract_cmd):
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
else:
    logger.warning(f"Tesseract executable not found at {tesseract_cmd}. OCR may fail.")

# --- Helper Function to Read Files ---
def read_document(uploaded_file) -> Optional[str]:
    """Reads text content from uploaded file (TXT, PDF, DOCX), with OCR for scanned PDFs."""
    try:
        # Limit file size to 10MB
        if uploaded_file.size > 10 * 1024 * 1024:
            st.error(f"File '{uploaded_file.name}' is too large. Maximum size is 10MB.")
            return None

        if uploaded_file.type == "text/plain":
            return uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            text = ""
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    else:
                        # Try OCR for scanned PDFs
                        try:
                            img = page.to_image().original
                            text += pytesseract.image_to_string(img) + "\n"
                        except Exception as ocr_e:
                            logger.warning(f"OCR failed for page in '{uploaded_file.name}': {ocr_e}")
                            st.warning(f"OCR failed for page in '{uploaded_file.name}': {ocr_e}")
            return text.strip() if text.strip() else None
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text.strip() if text.strip() else None
        else:
            st.error(f"Unsupported file type: {uploaded_file.type}")
            return None
    except Exception as e:
        logger.error(f"Error reading file '{uploaded_file.name}': {e}")
        st.error(f"Error reading file '{uploaded_file.name}': {e}")
        return None

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="Legal Document Analyzer")
st.title("📄 Legal Document Analysis Engine")
st.markdown("Upload your legal document (TXT, PDF, or DOCX) to get insights.")

# --- Initialize Session State ---
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'uploaded_filename' not in st.session_state:
    st.session_state.uploaded_filename = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None

# Reset button to clear session state
if st.button("Reset Analysis"):
    st.session_state.analysis_results = None
    st.session_state.uploaded_filename = None
    st.session_state.error_message = None
    st.info("Session state cleared. Upload a new file to start.")

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "Choose a document file",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=False
)

# --- Document Processing and Analysis Trigger ---
if uploaded_file is not None:
    # Process new file if different from previous
    if uploaded_file.name != st.session_state.uploaded_filename:
        st.session_state.analysis_results = None
        st.session_state.error_message = None
        st.session_state.uploaded_filename = uploaded_file.name
        with st.spinner(f"Reading '{uploaded_file.name}'..."):
            document_text = read_document(uploaded_file)
            if document_text:
                st.success(f"Successfully read '{uploaded_file.name}'. Ready for analysis.")
            else:
                st.error(f"Could not read text from '{uploaded_file.name}'.")
                document_text = None
    else:
        # Re-use existing document text if analysis was already done
        document_text = None if st.session_state.analysis_results is None else st.session_state.analysis_results.get("document_text")

    # Button to trigger analysis
    can_analyze = bool(uploaded_file and st.session_state.uploaded_filename == uploaded_file.name)
    if st.button("Analyze Document", disabled=not can_analyze):
        if st.session_state.analysis_results is None:
            if document_text is None:
                with st.spinner(f"Reading '{uploaded_file.name}' again..."):
                    document_text = read_document(uploaded_file)
            
            if document_text:
                if len(document_text) > 100000:
                    st.error("Document text is too long. Please upload a shorter document.")
                else:
                    st.session_state.error_message = None
                    with st.spinner("Analyzing document... This may take a minute or two."):
                        try:
                            results: Dict[str, Any] = analyze_legal_document(document_text)
                            if "error" in results:
                                st.session_state.error_message = results["error"]
                                st.error(f"Analysis failed: {results['error']}")
                            else:
                                st.session_state.analysis_results = results
                                logger.info("Analysis complete for '%s'", uploaded_file.name)
                                st.success("Analysis complete!")
                        except Exception as e:
                            logger.error(f"Analysis error: {e}")
                            st.session_state.error_message = f"An error occurred during analysis: {e}"
                            st.error(st.session_state.error_message)
            else:
                st.error(f"Could not read '{uploaded_file.name}' before analysis.")
        else:
            st.info("Analysis results are already displayed for this file.")

# --- Display Results ---
if st.session_state.error_message:
    st.error(f"Analysis failed: {st.session_state.error_message}")

if st.session_state.analysis_results:
    results = st.session_state.analysis_results
    if "error" not in results:
        st.divider()
        st.header("📊 Analysis Results")

        # 1. Summary Section
        st.subheader("Executive Summary")
        summary = results.get("summary", "Summary not available.")
        st.markdown(summary)

        st.divider()

        # Create columns for better layout of details
        col1, col2 = st.columns(2)

        with col1:
            # 2. Document Classification & Basic Info
            st.subheader("Document Details")
            doc_type = results.get("document_type", "N/A")
            confidence = results.get("confidence")
            jurisdiction = results.get("jurisdiction", "N/A")
            st.write(f"**Document Type:** {doc_type.replace('_', ' ').title() if doc_type else 'N/A'}")
            st.write(f"**Classification Confidence:** {f'{confidence:.1%}' if confidence is not None else 'N/A'}")
            st.write(f"**Detected Jurisdiction:** {jurisdiction}")

            # 3. Key Entities
            st.subheader("Key Entities")
            entities = results.get("entities")
            if isinstance(entities, dict):
                if "error" in entities:
                    st.warning(f"Entity Extraction Error: {entities['error']}")
                elif entities:
                    st.json(entities, expanded=False)
                else:
                    st.info("No specific entities extracted.")
            else:
                st.info("Entity data not available or in unexpected format.")

        with col2:
            # 4. Risk Analysis
            st.subheader("Risk Analysis")
            risk_analysis = results.get("risk_analysis")
            if isinstance(risk_analysis, dict):
                if "error" in risk_analysis:
                    st.warning(f"Risk Analysis Error: {risk_analysis['error']}")
                else:
                    score = risk_analysis.get("overall_risk_score", "N/A")
                    summary = risk_analysis.get("risk_summary", "No risk summary provided.")
                    risks = risk_analysis.get("risks", [])

                    st.metric(label="Overall Risk Score", value=f"{score}/10" if isinstance(score, (int, float)) else score)
                    st.markdown("**Risk Summary:**")
                    st.write(summary)

                    if risks:
                        with st.expander("Show Detailed Risks", expanded=False):
                            try:
                                df_risks = pd.DataFrame(risks)
                                cols_order = ["severity", "description", "implication", "mitigation_suggestion"]
                                df_risks = df_risks[[col for col in cols_order if col in df_risks.columns]]
                                st.dataframe(df_risks, use_container_width=True)
                            except Exception as e:
                                logger.warning(f"Could not display detailed risks as table: {e}")
                                st.warning(f"Could not display detailed risks as table: {e}")
                                st.json(risks)
                    else:
                        st.info("No specific risks identified in the analysis.")
            else:
                st.info("Risk analysis data not available or in unexpected format.")

        st.divider()

        # 5. Obligations, Rights, Requirements
        st.subheader("Extracted Clauses")
        with st.expander("Obligations (What parties MUST do)", expanded=False):
            obligations = results.get("obligations", [])
            if obligations:
                try:
                    df_obligations = pd.DataFrame(obligations)
                    cols_order = ["severity", "party", "description"]
                    df_obligations = df_obligations[[col for col in cols_order if col in df_obligations.columns]]
                    st.dataframe(df_obligations, use_container_width=True)
                except Exception as e:
                    logger.warning(f"Could not display obligations as table: {e}")
                    st.warning(f"Could not display obligations as table: {e}")
                    st.json(obligations)
            else:
                st.info("No specific obligations extracted.")

        with st.expander("Rights (What parties CAN do)", expanded=False):
            rights = results.get("rights", [])
            if rights:
                try:
                    df_rights = pd.DataFrame(rights)
                    cols_order = ["party", "description"]
                    df_rights = df_risks[[col for col in cols_order if col in df_risks.columns]]
                    st.dataframe(df_rights, use_container_width=True)
                except Exception as e:
                    logger.warning(f"Could not display rights as table: {e}")
                    st.warning(f"Could not display rights as table: {e}")
                    st.json(rights)
            else:
                st.info("No specific rights extracted.")

        with st.expander("Requirements (Conditions to be met)", expanded=False):
            requirements = results.get("requirements", [])
            if requirements:
                try:
                    df_requirements = pd.DataFrame(requirements)
                    cols_order = ["description", "context"]
                    df_requirements = df_requirements[[col for col in cols_order if col in df_requirements.columns]]
                    st.dataframe(df_requirements, use_container_width=True)
                except Exception as e:
                    logger.warning(f"Could not display requirements as table: {e}")
                    st.warning(f"Could not display requirements as table: {e}")
                    st.json(requirements)
            else:
                st.info("No specific requirements extracted.")

        st.divider()

        # 6. Precedent Analysis & Suggestions
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Precedent Analysis")
            precedent_analysis = results.get("precedent_analysis")
            if isinstance(precedent_analysis, dict):
                if "error" in precedent_analysis:
                    st.warning(f"Precedent Analysis Error: {precedent_analysis['error']}")
                elif precedent_analysis.get("status") == "skipped":
                    st.info(f"Precedent analysis skipped: {precedent_analysis.get('reason')}")
                else:
                    st.markdown("**Standard Clauses Present:**")
                    st.write(precedent_analysis.get("standard_clauses_present", ["N/A"]))
                    st.markdown("**Potentially Missing Standard Clauses:**")
                    st.write(precedent_analysis.get("standard_clauses_potentially_missing", ["N/A"]))
                    st.markdown("**Non-Standard/Unusual Clauses:**")
                    st.write(precedent_analysis.get("non_standard_or_unusual_clauses", ["N/A"]))
                    st.markdown("**Deviation Summary:**")
                    st.write(precedent_analysis.get("deviation_analysis", "N/A"))
            else:
                st.info("Precedent analysis data not available or in unexpected format.")

        with col4:
            st.subheader("Improvement Suggestions")
            suggestions = results.get("improvement_suggestions", [])
            if suggestions:
                if len(suggestions) == 1 and suggestions[0].get("issue_identified", "").startswith("Generation Error"):
                    st.warning(f"Suggestion Generation Error: {suggestions[0].get('rationale', 'Unknown error')}")
                else:
                    try:
                        df_suggestions = pd.DataFrame(suggestions)
                        cols_order = ["priority", "issue_identified", "recommendation", "rationale"]
                        df_suggestions = df_suggestions[[col for col in cols_order if col in df_suggestions.columns]]
                        st.dataframe(df_suggestions, use_container_width=True)
                    except Exception as e:
                        logger.warning(f"Could not display suggestions as table: {e}")
                        st.warning(f"Could not display suggestions as table: {e}")
                        st.json(suggestions)
            else:
                st.info("No specific improvement suggestions generated.")

st.divider()
st.caption("Powered by LangGraph, Legal-BERT, and T5")