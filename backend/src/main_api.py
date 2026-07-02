import os
import json
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util
from google import genai

load_dotenv()

app = FastAPI(title="Enterprise AI Hybrid Candidate Discovery Suite")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.dirname(CURRENT_DIR)
default_jsonl_path = os.path.join(BACKEND_ROOT, "data", "candidates.jsonl")

try:
    from .interview_ai import generate_custom_interview_questions
except ImportError:
    from interview_ai import generate_custom_interview_questions


print("🔄 Loading Transformer Embedding Layer (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Transformer System Is Active.")


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return None, "GEMINI_API_KEY not found. Check your .env file."

    try:
        client = genai.Client(api_key=api_key)
        return client, None
    except Exception as e:
        return None, str(e)


def calculate_bm25_lexical_score(text: str, target_keywords: List[str]) -> float:
    score = 0.0
    words = text.split()
    doc_len = len(words) if len(words) > 0 else 1

    for keyword in target_keywords:
        tf = words.count(keyword)
        if tf > 0:
            score += (tf * 1.5) / (
                tf + 1.5 * (0.25 + 0.75 * (doc_len / 100.0))
            )

    return round(score, 2)


def process_candidate_lines_fast(
    lines: List[str],
    target_keywords: List[str],
    target_cities: List[str],
    query_embedding,
    top_k: int
):
    parsed_candidates = []
    corpus_texts_for_embedding = []

    MAX_DEMO_LIMIT = 400

    for line in lines:
        if len(parsed_candidates) >= MAX_DEMO_LIMIT:
            break

        if not line.strip():
            continue

        try:
            cand = json.loads(line)
            profile = cand.get("profile", {})

            headline_original = profile.get("headline", "No Headline Provided")
            headline = headline_original.lower()
            summary = profile.get("summary", "").lower()[:200]
            full_text = f"{headline} {summary}"

            parsed_candidates.append((cand, full_text, headline_original))
            corpus_texts_for_embedding.append(f"{headline} {summary[:100]}")

        except Exception:
            continue

    if not parsed_candidates:
        return []

    print(f"🧬 Batch Encoding on {len(corpus_texts_for_embedding)} profiles...")

    corpus_embeddings = embedding_model.encode(
        corpus_texts_for_embedding,
        convert_to_tensor=True,
        show_progress_bar=False
    )

    cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

    ranked_results = []

    for idx, (cand, full_text, headline_original) in enumerate(parsed_candidates):
        lexical_score = calculate_bm25_lexical_score(full_text, target_keywords)

        semantic_sim_score = round(float(cosine_scores[idx]) * 5.0, 2)

        city_score = 0.0
        matched_cities = []

        for city in target_cities:
            if city in full_text:
                city_score += 2.0
                matched_cities.append(city.capitalize())

        total_score = round(lexical_score + semantic_sim_score + city_score, 2)

        matched_skills = [k for k in target_keywords if k in full_text]
        missing_skills = [k for k in target_keywords if k not in full_text]

        coverage = round(
            (len(matched_skills) / max(len(target_keywords), 1)) * 100,
            1
        )

        reasons = []

        if matched_skills:
            reasons.append(f"Skills: {', '.join(matched_skills).upper()}")

        if missing_skills:
            reasons.append(f"Gaps: {', '.join(missing_skills).upper()}")

        if matched_cities:
            reasons.append(f"Location Match: {', '.join(matched_cities)}")

        if semantic_sim_score > 2.2:
            reasons.append("Strong Transformer semantic match")

        reason_string = " | ".join(reasons)

        if coverage >= 75 and total_score >= 5.0:
            suitability = "High"
        elif coverage >= 40 or total_score >= 3.0:
            suitability = "Medium"
        else:
            suitability = "Low"

        ranked_results.append({
            "Candidate ID": cand.get("candidate_id", "UNKNOWN"),
            "Headline": headline_original,
            "Lexical Score (BM25)": lexical_score,
            "Semantic Similarity (Transformer)": semantic_sim_score,
            "Location Bonus": city_score,
            "Total Hybrid Match Score": total_score,
            "Matched Skills": matched_skills,
            "Missing Skills": missing_skills,
            "Skill Coverage %": coverage,
            "Suitability Level": suitability,
            "AI Suitability Reason": reason_string
        })

    ranked_results.sort(
        key=lambda x: x["Total Hybrid Match Score"],
        reverse=True
    )

    return ranked_results[:top_k]


@app.get("/")
def home():
    return {
        "message": "Enterprise AI Hybrid Candidate Discovery Suite backend running"
    }


@app.post("/rank_dynamic")
async def rank_candidates_dynamic(
    keywords: str = Form(...),
    cities: str = Form(""),
    top_k: int = Form(10),
    custom_jd: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    target_keywords = [
        k.strip().lower()
        for k in keywords.split(",")
        if k.strip()
    ]

    target_cities = [
        c.strip().lower()
        for c in cities.split(",")
        if c.strip()
    ]

    query_string = f"{' '.join(target_keywords)} {' '.join(target_cities)}"

    if custom_jd:
        query_string += f" {custom_jd.lower()}"

    query_embedding = embedding_model.encode(
        query_string,
        convert_to_tensor=True
    )

    if file:
        contents = await file.read()
        lines = contents.decode("utf-8").splitlines()
    else:
        if os.path.exists(default_jsonl_path):
            with open(default_jsonl_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        else:
            return {
                "error": f"Dataset file missing at {default_jsonl_path}"
            }

    return process_candidate_lines_fast(
        lines,
        target_keywords,
        target_cities,
        query_embedding,
        top_k
    )


@app.get("/interview")
def get_interview_questions(
    headline: str = "",
    matched: str = "",
    missing: str = ""
):
    matched_list = [m.strip() for m in matched.split(",") if m.strip()]
    missing_list = [mi.strip() for mi in missing.split(",") if mi.strip()]

    questions = generate_custom_interview_questions(
        headline,
        matched_list,
        missing_list
    )

    return {
        "interview_script": questions
    }


@app.post("/generate_questions_online")
async def generate_questions_online(
    role_headline: str = Form(...),
    required_skills: str = Form(...),
    job_description: str = Form("")
):
    client, error = get_gemini_client()

    if error:
        return {
            "error": error
        }

    prompt = f"""
Create a professional online assessment paper.

Role: {role_headline}

Required Skills:
{required_skills}

Job Description:
{job_description}

Include:

1. Exam Overview
- Duration
- Total questions
- Section breakdown

2. Technical MCQs
Create 5 high-quality MCQs.
Each MCQ must include:
- Question
- A, B, C, D options
- Correct answer
- Short explanation

3. Coding Questions
Create 3 coding/design questions.

4. Practical Tasks
Create 2 practical project-style tasks.

5. Evaluation Criteria
Give marking criteria out of 100.

Use clean Markdown format.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        assessment_sheet = getattr(response, "text", None)

        if assessment_sheet:
            return {
                "assessment_sheet": assessment_sheet
            }

        return {
            "assessment_sheet": fallback_assessment(
                role_headline,
                required_skills,
                job_description,
                "Gemini returned empty response."
            )
        }

    except Exception as e:
        return {
            "assessment_sheet": fallback_assessment(
                role_headline,
                required_skills,
                job_description,
                str(e)
            )
        }


def fallback_assessment(
    role_headline: str,
    required_skills: str,
    job_description: str,
    error_message: str
) -> str:
    return f"""
# Dynamic Assessment Paper

## Role
{role_headline}

## Required Skills
{required_skills}

## Job Description
{job_description}

---

## 1. Exam Overview

- Duration: 60 minutes
- Total Questions: 10
- Total Marks: 100

---

## 2. Technical MCQs

1. What is the main purpose of FastAPI?
   - A. Frontend styling
   - B. API development in Python
   - C. Database backup
   - D. Image editing

   Correct Answer: B

2. Why is Docker used?
   - A. To containerize applications
   - B. To write CSS
   - C. To replace Python
   - D. To create images only

   Correct Answer: A

3. What is REST API?
   - A. A web API design style
   - B. A database
   - C. A cloud server
   - D. A programming language

   Correct Answer: A

4. What is AWS commonly used for?
   - A. Cloud services
   - B. Mobile charging
   - C. Offline editing
   - D. Gaming only

   Correct Answer: A

5. What is React mainly used for?
   - A. Building user interfaces
   - B. Training AI models only
   - C. Writing backend APIs
   - D. Managing files

   Correct Answer: A

---

## 3. Coding Questions

1. Create a FastAPI endpoint that returns a list of candidates.
2. Write Python code to filter candidates based on matching skills.
3. Create a basic React component to display candidate details.

---

## 4. Practical Tasks

1. Build a mini candidate ranking API.
2. Containerize the API using Docker.

---

## 5. Evaluation Criteria

- Technical Knowledge: 30 marks
- Coding Ability: 30 marks
- System Design: 20 marks
- Explanation Quality: 20 marks

---

## Fallback Note

Gemini response was unavailable, so this fallback assessment was generated.

Backend message:
{error_message}
"""
import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.src.main_api:app", host="0.0.0.0", port=port)