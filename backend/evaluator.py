import openai
from utils import chunk_text

import os

openai.api_key = os.getenv("OPENAI_API_KEY")

# -------------------------------
# 1. Generate Logic-Based Questions
# -------------------------------

def generate_logic_questions(document_text, num_questions=3):
    """
    Uses OpenAI to generate logic-based or comprehension-heavy questions from the document.
    Returns a list of dicts: {question, ideal_answer, source_chunk}.
    """
    chunks = chunk_text(document_text, chunk_size=1500, chunk_overlap=200)
    context = chunks[0] if len(chunks) > 0 else document_text[:2000]  # Use first chunk

    prompt = (
        f"Based only on the document content below, generate {num_questions} "
        f"logic-based or comprehension-focused questions. "
        f"For each question, provide an ideal answer and the reference text that supports it.\n\n"
        f"Document:\n{context}\n\n"
        f"Format:\n"
        f"Q1: <question>\nA1: <ideal answer>\nRef1: <supporting snippet>\n..."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or gpt-4 if you prefer
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=700
        )
        content = response.choices[0].message.content.strip()
        return parse_generated_questions(content)

    except Exception as e:
        print("Error generating questions:", e)
        return []

# -------------------------------
# 2. Parse GPT Response into Dicts
# -------------------------------

def parse_generated_questions(gpt_output):
    """
    Parses GPT output string into a list of structured questions.
    """
    questions = []
    entries = gpt_output.strip().split("Q")
    for entry in entries:
        if not entry.strip():
            continue
        try:
            parts = entry.strip().split("\n")
            q = parts[0].split(":")[1].strip()
            a = parts[1].split(":")[1].strip()
            ref = parts[2].split(":", 1)[1].strip()
            questions.append({
                "question": q,
                "ideal_answer": a,
                "source_chunk": ref
            })
        except Exception as e:
            print("Error parsing question entry:", e)
            continue
    return questions

# -------------------------------
# 3. Evaluate User Answer
# -------------------------------

def evaluate_answer(question, user_answer, ideal_answer, source_chunk):
    """
    Compares user answer to the ideal answer and provides feedback.
    Justifies the feedback using the source_chunk.
    """
    prompt = (
        f"You are a strict academic evaluator. Compare the user's answer to the ideal answer.\n\n"
        f"Question: {question}\n"
        f"Ideal Answer: {ideal_answer}\n"
        f"User's Answer: {user_answer}\n\n"
        f"Evaluate if the user's answer is correct, partially correct, or incorrect. "
        f"Give a 1–2 sentence explanation. Do not hallucinate. Base your explanation only on the document.\n"
        f"Reference:\n{source_chunk}"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200
        )
        feedback = response.choices[0].message.content.strip()
        return {
            "feedback": feedback,
            "justification": source_chunk
        }

    except Exception as e:
        return {
            "feedback": f"Error during evaluation: {str(e)}",
            "justification": source_chunk
        }
