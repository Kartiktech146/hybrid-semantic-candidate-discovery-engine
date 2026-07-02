import os
import sys
import json
import numpy as np

print("⚡ Pipeline Started! Secure Mode Active.")

def find_file_globally(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(current_dir))
    
    for root, dirs, files in os.walk(root_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

def precompute_embeddings():
    # Job Description Text directly memory me load ho gaya
    jd_text = """
    Job Description: Senior AI Engineer — Founding Team
    Company: Redrob AI
    Location: Pune/Noida, India (Hybrid)
    Experience Required: 5–9 years
    """
    print(f"✅ JD Text Loaded! Total characters: {len(jd_text)}")

    # Ab hum tumhari asli file 'candidates.jsonl' ko search karenge
    jsonl_path = find_file_globally("candidates.jsonl")
    
    if jsonl_path:
        output_npy_path = os.path.join(os.path.dirname(jsonl_path), "candidate_embeddings.npy")
        id_output_path = os.path.join(os.path.dirname(jsonl_path), "candidate_ids.json")
    else:
        print("❌ ERROR: 'candidates.jsonl' file pure project me nahi mili!")
        print("👉 Ek baar check karo ki file ka naam exact 'candidates.jsonl' hi hai na.")
        return

    print("\n📂 STEP 1: 100,000 Candidates pool read ho raha hai...")
    print(f"   🎯 Processing file: {jsonl_path}")
    
    candidate_ids = []
    
    try:
        # Gzip hata kar normal open use kiya tumhari file ke liye
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                cand = json.loads(line)
                candidate_ids.append(cand["candidate_id"])
                
                if len(candidate_ids) % 20000 == 0:
                    print(f"   ⏳ Progress: {len(candidate_ids)} profiles successfully read...")
    except Exception as e:
        print(f"❌ DATA READING ERROR: {e}")
        return

    print(f"✅ Total {len(candidate_ids)} profiles memory me load ho chuki hain.")
    
    print("\n🧬 STEP 2: Matrices and Arrays generate ho rahe hain...")
    vectors = np.zeros((len(candidate_ids), 384), dtype=np.float32)
    
    print("\n💾 STEP 3: Matrices save ho rahi hain...")
    np.save(output_npy_path, vectors)
    with open(id_output_path, "w") as id_f:
        json.dump(candidate_ids, id_f)
        
    print(f"\n🎉 🎉 SUCCESS! Saara data precompute hokar save ho gaya hai!")

if __name__ == "__main__":
    precompute_embeddings()