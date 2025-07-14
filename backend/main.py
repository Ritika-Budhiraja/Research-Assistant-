# main.py — Orchestrates backend logic

from backend.utils import extract_text_from_file
from backend.summarizer import generate_summary
from backend.qa_engine import build_vectorstore, answer_query
from backend.evaluator import generate_logic_questions, evaluate_answer

class SmartAssistant:
    def __init__(self):
        self.document_text = None
        self.vectorstore = None
        self.summary = None
        self.logic_questions = None

    def load_document(self, file, file_ext):
        """Extract text and initialize vectorstore."""
        self.document_text = extract_text_from_file(file, file_ext)
        self.vectorstore = build_vectorstore(self.document_text)
        return self.document_text

    def summarize_document(self):
        """Generate concise summary."""
        if not self.document_text:
            return "No document loaded."
        self.summary = generate_summary(self.document_text)
        return self.summary

    def handle_question(self, query):
        """Answer user query from document."""
        if not self.vectorstore:
            return {"answer": "Document not loaded.", "source": ""}
        return answer_query(query, self.vectorstore)

    def generate_challenge_questions(self):
        """Generate 3 logic-based questions from the doc."""
        if not self.document_text:
            return []
        self.logic_questions = generate_logic_questions(self.document_text)
        return self.logic_questions

    def evaluate_user_response(self, question, user_answer, ideal_answer, source_chunk):
        """Evaluate user response and return feedback with justification."""
        return evaluate_answer(question, user_answer, ideal_answer, source_chunk)
