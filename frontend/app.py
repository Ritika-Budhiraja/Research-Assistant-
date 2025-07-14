import streamlit as st
from dotenv import load_dotenv
import openai
import os
import sys

# 🛠 Add backend folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

# 🔐 Load OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 🧠 Import backend logic
from utils import extract_text_from_file
from summarizer import generate_summary
from qa_engine import build_vectorstore, answer_query
from evaluator import generate_logic_questions, evaluate_answer

# 🎯 Page Config
st.set_page_config(page_title="Smart Research Assistant", layout="wide")
st.title("📄 Smart Research Assistant")
st.markdown("Upload a research paper or document (PDF/TXT), and explore its content using GenAI.")

# 💾 Session State Initialization
if "document_text" not in st.session_state:
    st.session_state.document_text = ""
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "questions" not in st.session_state:
    st.session_state.questions = []

# 📎 File Upload
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    file_ext = os.path.splitext(uploaded_file.name)[1]
    with st.spinner("📖 Reading and understanding the document..."):
        try:
            text = extract_text_from_file(uploaded_file, file_ext)
            st.session_state.document_text = text
            st.session_state.vectorstore = build_vectorstore(text)   # FAISS object with as_retriever()
            st.session_state.summary = generate_summary(text)
            st.session_state.questions = generate_logic_questions(text)
            st.success("✅ Document processed successfully!")
        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")

    # 🔍 Show Summary
    if st.session_state.summary:
        st.subheader("📌 Summary (≤150 words):")
        st.write(st.session_state.summary)

    # 🚦 Mode Switch
    mode = st.radio("Choose interaction mode:", ["Ask Anything", "Challenge Me"])

    # 💬 Mode 1: Ask Anything
    if mode == "Ask Anything":
        st.subheader("💬 Ask Anything from the Document")
        user_query = st.text_input("Enter your question here:")

        if user_query:
            with st.spinner("Generating answer..."):
                try:
                    response = answer_query(user_query, st.session_state.vectorstore)
                    st.markdown("#### ✅ Answer:")
                    st.write(response["answer"])
                    st.markdown("#### 📍 Justification:")
                    st.write(response["source"])
                except Exception as e:
                    st.error(f"❌ Failed to generate answer: {str(e)}")

    # 🧠 Mode 2: Challenge Me
    elif mode == "Challenge Me":
        st.subheader("🧠 Logic-Based Questions")
        for i, q in enumerate(st.session_state.questions):
            st.markdown(f"**Q{i+1}: {q['question']}**")
            user_answer = st.text_input("Your Answer:", key=f"answer_{i}")
            if user_answer:
                with st.spinner("Evaluating your answer..."):
                    feedback = evaluate_answer(
                        q["question"],
                        user_answer,
                        q["ideal_answer"],
                        q["source_chunk"]
                    )
                st.markdown("✅ **Feedback:**")
                st.write(feedback["feedback"])
                st.markdown("📍 **Justification:**")
                st.write(feedback["justification"])

else:
    st.info("📁 Please upload a document to begin.")
