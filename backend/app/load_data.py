import json
import gzip

# Define absolute deal-breakers for an AI Engineering role
INVALID_ROLES = ["accountant", "marketing manager", "hr manager", "sales executive", "recruiter"]

def clean_and_filter_candidates(file_path):
    valid_candidates = []
    
    # Open the compressed jsonl file safely
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            candidate = json.loads(line)
            profile = candidate.get("profile", {})
            current_title = profile.get("current_title", "").lower()
            
            # TRAP DETECTION: If their current role is completely unrelated, skip them!
            if any(role in current_title for role in INVALID_ROLES):
                continue
                
            # If they pass the trap check, keep them for scoring
            valid_candidates.append(candidate)
            
    return valid_candidates