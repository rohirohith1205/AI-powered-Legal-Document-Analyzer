from typing import Dict, List, Optional, Any, TypedDict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import pipeline
import re
from langchain_core.documents import Document
from datetime import datetime
import os
import logging
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langgraph")
from langgraph.graph import StateGraph, END

try:
    from contract_analysis_rag import ContractAnalysisRAG
    from mychunk import Chunk
except ImportError as e:
    raise ImportError(f"Failed to import ContractAnalysisRAG or Chunk: {e}. Ensure 'contract_analysis_rag.py' and 'mychunk.py' are in the project directory.")

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load .env file
try:
    from dotenv import load_dotenv
    dotenv_path = os.getenv("DOTENV_PATH", os.path.join(os.path.dirname(__file__), ".env"))
    load_dotenv(dotenv_path=dotenv_path)
except ImportError:
    logger.warning("dotenv not installed. Environment variables must be set manually.")

# Initialize models
try:
    encoder = SentenceTransformer('nlpaueb/legal-bert-base-uncased')
    summarizer = pipeline("summarization", model="t5-small")
except Exception as e:
    logger.error(f"Failed to load Legal-BERT or T5 model: {e}")
    raise

# Initialize RAG system globally but allow reset
rag_system = ContractAnalysisRAG(use_semantic_search=True)

class GraphState(TypedDict):
    """Represents the state of the legal document analysis graph."""
    document_text: str
    document_type: Optional[str]
    confidence: Optional[float]
    jurisdiction: Optional[str]
    documents: Optional[List[Document]]
    total_chunks: Optional[int]
    original_length: Optional[int]
    entities: Optional[Dict[str, Any]]
    obligations: Optional[List[Dict[str, str]]]
    rights: Optional[List[Dict[str, str]]]
    requirements: Optional[List[Dict[str, str]]]
    risk_analysis: Optional[Dict[str, Any]]
    precedent_analysis: Optional[Dict[str, Any]]
    improvement_suggestions: Optional[List[Dict[str, str]]]
    summary: Optional[str]

def document_classifier(state: GraphState) -> GraphState:
    """Classify the document type and detect jurisdiction."""
    document_text = state["document_text"]
    try:
        doc_embedding = encoder.encode([document_text])[0]
        predefined_types = ["rental_agreement", "contract", "lease", "nda"]
        type_embeddings = encoder.encode(predefined_types, batch_size=32)
        similarities = cosine_similarity([doc_embedding], type_embeddings)[0]
        max_idx = np.argmax(similarities)
        state["document_type"] = predefined_types[max_idx]
        state["confidence"] = float(similarities[max_idx])
        # Basic jurisdiction detection
        jurisdiction_keywords = {
            "california": "California",
            "new york": "New York",
            "texas": "Texas",
        }
        text_lower = document_text.lower()
        for keyword, jurisdiction in jurisdiction_keywords.items():
            if keyword in text_lower:
                state["jurisdiction"] = jurisdiction
                break
        else:
            state["jurisdiction"] = "Unknown"
        return state
    except Exception as e:
        logger.error(f"Document classification error: {e}")
        return {
            **state,
            "document_type": "error",
            "confidence": 0.0,
            "jurisdiction": "Unknown"
        }

def document_processor(state: GraphState) -> GraphState:
    """Process the document into chunks using the RAG system."""
    document_text = state["document_text"]
    try:
        # Reset RAG chunks to avoid accumulating old data
        rag_system.chunks = []
        rag_system.embeddings = None
        documents = [{"text": document_text, "metadata": {"source": "uploaded_document", "timestamp": datetime.now().isoformat()}}]
        rag_system.add_documents(documents)
        state["documents"] = [Document(page_content=doc["text"], metadata=doc["metadata"]) for doc in documents]
        state["total_chunks"] = len(rag_system.chunks)
        state["original_length"] = len(document_text)
        return state
    except Exception as e:
        logger.error(f"Document processing error: {e}")
        return {**state, "documents": None, "total_chunks": 0, "original_length": len(document_text)}

def entity_extractor(state: GraphState) -> GraphState:
    """Extract key entities like parties, dates, and amounts from the document."""
    document_text = state["document_text"]
    try:
        entities = {}
        patterns = {
            "parties": r"(?:between|by and between)\s+([A-Za-z\s,]+?)\s+and\s+([A-Za-z\s,]+?)(?=\s*(?:,|\.|;|$))",
            "dates": r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b",
            "amounts": r"\$\d+(?:,\d{3})*(?:\.\d{2})?\b"
        }
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, document_text, re.IGNORECASE)
            entities[entity_type] = matches
        return {**state, "entities": entities}
    except Exception as e:
        logger.error(f"Entity extraction error: {e}")
        return {**state, "entities": {"error": str(e)}}

def obligation_extractor(state: GraphState) -> GraphState:
    """Extract obligations, rights, and requirements from the document."""
    document_text = state["document_text"]
    try:
        obligations = []
        obligation_patterns = [
            (r"shall\s+pay\s+.*?\$(\d+(?:,\d{3})*(?:\.\d{2})?)", "payment", "high"),
            (r"shall\s+provide\s+.*?within\s+(\d+)\s+days", "action", "medium"),
            (r"must\s+maintain\s+.*?(?:insurance|condition)", "maintenance", "medium")
        ]
        for pattern, desc, severity in obligation_patterns:
            matches = re.finditer(pattern, document_text, re.IGNORECASE)
            for match in matches:
                parties = state["entities"].get("parties", [])
                party = parties[0][1] if parties else "Unknown"
                obligations.append({
                    "description": match.group(0),
                    "party": party,
                    "severity": severity
                })
        rights = [
            {"party": party, "description": f"Right to terminate with notice"} 
            for party in (state["entities"].get("parties", [])[0] if state["entities"].get("parties") else [])
        ]
        requirements = [
            {"description": match.group(0), "context": "Contractual condition"}
            for match in re.finditer(r"subject\s+to\s+.*?(?:approval|condition)", document_text, re.IGNORECASE)
        ]
        return {**state, "obligations": obligations, "rights": rights, "requirements": requirements}
    except Exception as e:
        logger.error(f"Obligation extraction error: {e}")
        return {**state, "obligations": [], "rights": [], "requirements": []}

def risk_analyzer(state: GraphState) -> GraphState:
    """Analyze risks based on extracted obligations, missing clauses, and critical terms."""
    try:
        obligations = state.get("obligations", []) or []
        precedent = state.get("precedent_analysis", {})
        missing_clauses = precedent.get("standard_clauses_potentially_missing", []) if precedent else []

        risk_score = 0
        risks = []

        # Evaluate obligations
        for obl in obligations:
            severity = obl.get("severity", "low")
            desc = obl.get("description", "")
            if severity == "high":
                risk_score += 3
            elif severity == "medium":
                risk_score += 2
            else:
                risk_score += 1

            risks.append({
                "severity": severity,
                "description": desc,
                "implication": "Potential legal or financial exposure.",
                "mitigation_suggestion": "Ensure obligations are clearly defined and achievable."
            })

        # Add points for missing standard clauses
        for clause in missing_clauses:
            risk_score += 1
            risks.append({
                "severity": "medium",
                "description": f"Missing standard clause: {clause}",
                "implication": "Missing standard clauses can increase ambiguity or disputes.",
                "mitigation_suggestion": f"Consider adding a '{clause}' clause for better clarity."
            })

        # Keyword-based checks for risky terms
        text = state["document_text"].lower()
        keywords = {
            "penalty": 3,
            "liability": 3,
            "indemnity": 3,
            "termination": 2,
            "breach": 2,
            "fine": 2,
        }
        for term, score in keywords.items():
            if term in text:
                risk_score += score
                risks.append({
                    "severity": "medium",
                    "description": f"Contains '{term}' term — review for fairness.",
                    "implication": f"The '{term}' clause could carry one-sided or heavy liabilities.",
                    "mitigation_suggestion": "Ensure balance between both parties."
                })

        # Normalize and cap risk score
        risk_score = min(risk_score, 10)
        risk_summary = f"Detected {len(risks)} potential risk indicators. Overall risk score: {risk_score}/10."

        return {
            **state,
            "risk_analysis": {
                "overall_risk_score": risk_score,
                "risk_summary": risk_summary,
                "risks": risks
            }
        }

    except Exception as e:
        logger.error(f"Risk analysis error: {e}")
        return {**state, "risk_analysis": {"error": str(e)}}


def precedent_analyzer(state: GraphState) -> GraphState:
    """Analyze the presence of standard clauses in the document."""
    try:
        standard_clauses = ["term", "rent", "security deposit", "termination", "governing law"]
        present_clauses = []
        missing_clauses = []
        document_text = state["document_text"].lower()
        for clause in standard_clauses:
            if re.search(rf"\b{clause}\b", document_text):
                present_clauses.append(clause)
            else:
                missing_clauses.append(clause)
        return {
            **state,
            "precedent_analysis": {
                "standard_clauses_present": present_clauses,
                "standard_clauses_potentially_missing": missing_clauses,
                "non_standard_or_unusual_clauses": [],
                "deviation_analysis": f"Missing {len(missing_clauses)} standard clauses."
            }
        }
    except Exception as e:
        logger.error(f"Precedent analysis error: {e}")
        return {**state, "precedent_analysis": {"error": str(e)}}

def suggestion_generator(state: GraphState) -> GraphState:
    """Generate improvement suggestions based on precedent analysis."""
    try:
        suggestions = []
        if state["precedent_analysis"]:
            for clause in state["precedent_analysis"].get("standard_clauses_potentially_missing", []):
                suggestions.append({
                    "priority": "medium",
                    "issue_identified": f"Missing {clause} clause",
                    "recommendation": f"Add a {clause} clause to align with standard contracts.",
                    "rationale": "Standard clauses reduce legal ambiguity."
                })
        return {**state, "improvement_suggestions": suggestions}
    except Exception as e:
        logger.error(f"Suggestion generation error: {e}")
        return {**state, "improvement_suggestions": [{"issue_identified": f"Generation Error: {str(e)}"}]}

def summary_generator(state: GraphState) -> GraphState:
    """Generate a summary of the document."""
    try:
        max_input_length = 1000
        text_to_summarize = state["document_text"][:max_input_length]
        summary = summarizer(text_to_summarize, max_length=150, min_length=30, do_sample=False)[0]["summary_text"]
        return {**state, "summary": summary}
    except Exception as e:
        logger.error(f"Summary generation error: {e}")
        return {**state, "summary": f"Summary generation failed: {str(e)}"}

def create_legal_analysis_graph():
    """Builds the LangGraph StateGraph for legal document analysis."""
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("document_classifier", document_classifier)
    workflow.add_node("document_processor", document_processor)
    workflow.add_node("entity_extractor", entity_extractor)
    workflow.add_node("obligation_extractor", obligation_extractor)
    workflow.add_node("risk_analyzer", risk_analyzer)
    workflow.add_node("precedent_analyzer", precedent_analyzer)
    workflow.add_node("suggestion_generator", suggestion_generator)
    workflow.add_node("summary_generator", summary_generator)

    # Define the execution flow using edges
    workflow.set_entry_point("document_classifier")
    workflow.add_edge("document_classifier", "document_processor")
    workflow.add_edge("document_processor", "entity_extractor")
    workflow.add_edge("entity_extractor", "obligation_extractor")
    workflow.add_edge("obligation_extractor", "risk_analyzer")
    workflow.add_edge("risk_analyzer", "precedent_analyzer")
    workflow.add_edge("precedent_analyzer", "suggestion_generator")
    workflow.add_edge("suggestion_generator", "summary_generator")
    workflow.add_edge("summary_generator", END)

    # Compile the graph
    logger.info("Compiling the graph...")
    app = workflow.compile()
    logger.info("Graph compiled successfully.")
    return app

def analyze_legal_document(document_text: str) -> Dict[str, Any]:
    """
    Analyze a legal document using the LangGraph workflow.
    
    Args:
        document_text (str): The text content of the legal document.
    
    Returns:
        Dict[str, Any]: The final state of the analysis, including all extracted information.
    
    Raises:
        ValueError: If document_text is invalid.
    """
    try:
        if not isinstance(document_text, str) or not document_text.strip():
            raise ValueError("Document text must be a non-empty string")
        if len(document_text) > 100000:
            raise ValueError("Document text is too long for processing")

        # Initialize the graph
        app = create_legal_analysis_graph()

        # Define the initial state
        initial_state: GraphState = {
            "document_text": document_text,
            "document_type": None,
            "confidence": None,
            "jurisdiction": None,
            "documents": None,
            "total_chunks": None,
            "original_length": None,
            "entities": None,
            "obligations": None,
            "rights": None,
            "requirements": None,
            "risk_analysis": None,
            "precedent_analysis": None,
            "improvement_suggestions": None,
            "summary": None,
        }

        # Run the graph
        logger.info("Running the analysis graph...")
        final_state = app.invoke(initial_state)
        logger.info("Analysis complete.")
        return final_state
    except Exception as e:
        logger.error(f"Error in analyze_legal_document: {e}")
        return {
            "error": str(e),
            "document_text": document_text,
            "summary": f"Analysis failed: {str(e)}"
        }