import openai
import os
from dotenv import load_dotenv
import os

load_dotenv()  # Load from .env file
openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_summary(text, word_limit=150):
    """
    Summarizes the input text using OpenAI GPT.
    Limits output to ~150 words.
    """
    prompt = (
        f"Summarize the following document in under {word_limit} words. "
        f"Ensure the summary is factual and only based on the content provided.\n\n"
        f"{text[:4000]}"  # Truncate to first 4000 characters to fit token limit
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if available
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error generating summary: {str(e)}"
