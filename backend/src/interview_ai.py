import os
import traceback
from google import genai
from dotenv import load_dotenv

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

# 🌟 GLOBAL CLIENT EXPORT (FastAPI ke imports ke liye zaroori hai)
try:
    print(f"🔑 Initializing Gemini Client Matrix with key prefix: {api_key[:5]}...")
    client = genai.Client(api_key=api_key)
except Exception as init_err:
    print(f"❌ Gemini initialization failure: {str(init_err)}")
    client = None


def generate_custom_interview_questions(headline: str, matched_skills: list, missing_skills: list) -> str:
    """
    Generates targeted technical interview questions for single candidates
    """
    if not client:
        return "⚠️ Gemini Engine Error: Client SDK initialization failed."

    try:
        prompt = f"""
        You are an expert technical interviewer. Analyze this candidate profile and generate exactly 5 personalized interview questions.
        
        Candidate Headline: {headline if headline else 'No Specific Headline'}
        Matched Core Skills: {', '.join(matched_skills).upper() if matched_skills else 'None'}
        Skill Gaps (Missing): {', '.join(missing_skills).upper() if missing_skills else 'None'}
        
        Structure Matrix:
        - 2 Technical questions exploring deeply into their Matched Skills.
        - 2 Adaptive questions to assess their ability to navigate their Missing Skills / Gaps.
        - 1 Complex architectural scenario question customized to their Headline role.
        
        Format the response beautifully using clean Markdown styling tags.
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        
        if response and response.text:
            return response.text
        return "⚠️ Generative response returned an empty payload."

    except Exception as runtime_err:
        print(f"🚨 Deep Trace Exception inside single candidate interviewer core:")
        traceback.print_exc()
        return f"⚠️ Generative AI Runtime Failure: {str(runtime_err)}"