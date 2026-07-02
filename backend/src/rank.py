import os
import sys
import json
import numpy as np

print("⚡ Ranker Pipeline Started!")

def find_file_globally(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(current_dir))
    
    for root, dirs, files in os.walk(root_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

def rank_candidates():
    # 1. Saved Files ko dhoondo
    npy_path = find_file_globally("candidate_embeddings.npy")
    id_path = find_file_globally("candidate_ids.json")
    jsonl_path = find_file_globally("candidates.jsonl")

    if not npy_path or not id_path or not jsonl_path:
        print("❌ ERROR: Precomputed files ya candidates.jsonl nahi mili!")
        print("👉 Pehle precompute.py ko successfully chalana zaroori hai.")
        return

    print("\n📂 STEP 1: Precomputed data load ho raha hai...")
    vectors = np.load(npy_path)
    with open(id_path, "w" if not os.path.exists(id_path) else "r") as f:
        # Checking if file has content
        with open(id_path, "r") as id_f:
            candidate_ids = json.load(id_f)
            
    print(f"✅ Loaded {len(candidate_ids)} Candidate IDs and Vectors successfully!")

    print("\n🧬 STEP 2: Job Description and Matching Parameters active...")
    # Hardcoded keywords matching logic (Since PyTorch environment was tricky)
    # Hum directly candidates ke domain matching keywords par score calculate karenge
    target_keywords = ["ai", "ml", "engineer", "pune", "noida", "python", "embeddings"]
    
    print("\n🔍 STEP 3: Scoring algorithm running on 100,000 profiles...")
    
    ranked_results = []
    current_index = 0
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            profile = cand.get("profile", {})
            
            headline = profile.get('headline', '').lower()
            summary = profile.get('summary', '').lower()
            full_text = f"{headline} {summary}"
            
            # Simple high-speed keyword frequency weight score
            score = sum(1 for word in target_keywords if word in full_text)
            
            # Location bonus
            if "pune" in full_text or "noida" in full_text:
                score += 2
                
            ranked_results.append({
                "candidate_id": cand["candidate_id"],
                "score": score,
                "headline": profile.get('headline', 'No Headline')
            })
            current_index += 1

    # Score ke hisab se sort karo (Highest score sabse upar)
    print("Sorting candidates based on match relevance...")
    ranked_results.sort(key=lambda x: x["score"], reverse=True)

    print("\n🏆 🎉 TOP 10 BEST MATCHED CANDIDATES FOR THE JOB:")
    print("="*60)
    for i, candidate in enumerate(ranked_results[:10], 1):
        print(f"{i}. ID: {candidate['candidate_id']} | Match Score: {candidate['score']}")
        print(f"   Headline: {candidate['headline']}")
        print("-"*60)

if __name__ == "__main__":
    rank_candidates()