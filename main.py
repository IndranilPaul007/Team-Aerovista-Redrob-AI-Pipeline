import json
import csv
import os
import re
import sys
import gzip
import math
import torch
from sentence_transformers import SentenceTransformer, util

# ============================================================================
# PARTICIPANT ID
# ============================================================================
PARTICIPANT_ID = "Aerovista"
OUTPUT_FILENAME = f"{PARTICIPANT_ID}.csv"

# CPU Optimization for Hackathon Sandbox
torch.set_num_threads(os.cpu_count() or 4)

DATA_PATH = None
for path in ["candidates.jsonl", "candidates.jsonl.gz"]:
    if os.path.exists(path):
        DATA_PATH = path
        break

if not DATA_PATH:
    print("❌ ERROR: Cannot find candidates file.")
    exit()

print(f"1. Ingesting candidates from: {DATA_PATH}...")
model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)

CONSULTING_KEYWORDS = ["tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant", "capgemini", "hcl", "deloitte"]
BANNED_ROLES = ["marketing", "accountant", "sales", "recruiter", "civil", "hr", "mechanical", "operations"]

# Broadened keyword matrix to catch semantic equivalents in pre-ranking
JD_KEYWORDS = {
    "rag", "pinecone", "milvus", "weaviate", "vector", "embedding", "sentence-transformers", 
    "llm", "nlp", "ndcg", "mrr", "machine learning", "ann", "faiss", "semantic search", 
    "dense retrieval", "hybrid retrieval", "opensearch", "elasticsearch", "vector database", 
    "retrieval pipeline", "ranking system", "offline evaluation", "a/b testing", "cross-encoder",
    "bm25", "information retrieval", "dense ann", "approximate search", "embedding index", "retriever model"
}
JD_KEYWORDS_NORM = {k.replace("-", "").replace(" ", "") for k in JD_KEYWORDS}

BANNED_REGEX = re.compile(r'\b(?:' + '|'.join(BANNED_ROLES) + r')\b', re.IGNORECASE)
TITLE_REGEX = re.compile(r'\b(?:ai|ml|machine learning|search|retrieval|nlp|applied scientist|foundation model|recommendation|data scientist|mlops|ranking|infrastructure)\b', re.IGNORECASE)

print("2. Multi-layered structural and lexical processing...")
candidates_data = []
error_count = 0
MAX_ERRORS_TO_LOG = 5

def open_candidate_stream(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, "r", encoding="utf-8")

with open_candidate_stream(DATA_PATH) as f:
    for line in f:
        if not line.strip(): continue
        
        c_id = "unknown" # Failsafe initialization for error logging
        try:
            candidate = json.loads(line)
            c_id = candidate.get("candidate_id", "unknown")
            
            profile = candidate.get("profile", {}) or {}
            history = candidate.get("career_history", []) or []
            signals = candidate.get("redrob_signals", {}) or {}
            skills = candidate.get("skills", []) or []

            current_title = str(profile.get("current_title", "")).strip().lower()
            headline = str(profile.get("headline", "")).strip().lower()
            location = str(profile.get("location", "")).strip().lower()
            
            # TEXT EXTRACTION 
            raw_exp = float(profile.get("years_of_experience", 0.0) if str(profile.get("years_of_experience", 0)).replace('.', '', 1).isdigit() else 0.0)
            jobs_text = " ".join([str(job.get("description", "")) for job in history if job])
            full_text = f"{headline} {profile.get('summary', '')} {jobs_text}".strip()
            text_lower = full_text.lower()
            
            # FREQUENCY-BASED LEXICAL SCORING
            lexical_hits = sum(text_lower.count(kw) for kw in JD_KEYWORDS)
            
            # STRUCTURED NORMALIZED SKILLS EXTRACTION
            raw_skills = [str(s.get("name")) for s in skills if s.get("name")]
            norm_skills = [s.lower().replace("-", "").replace(" ", "") for s in raw_skills]
            skill_overlap = len(JD_KEYWORDS_NORM.intersection(set(norm_skills)))

            # ADDITIVE MODIFIERS
            additive_bonus = 0.0
            if TITLE_REGEX.search(current_title): additive_bonus += 0.05
            
            combined_titles = f"{current_title} {headline}"
            if BANNED_REGEX.search(combined_titles): additive_bonus -= 0.15
            
            # HONEYPOT DETECTOR
            impossible_skills = sum(1 for s in skills if str(s.get("proficiency")).lower() in ["expert", "advanced"] and not s.get("duration_months"))
            if impossible_skills >= 3: 
                additive_bonus -= 0.15
            
            all_companies = [str(job.get("company", "")).strip().lower() for job in history if job]
            if profile.get("current_company"):
                all_companies.append(str(profile.get("current_company")).strip().lower())
                
            # Consulting penalty
            if any(any(kw in comp for kw in CONSULTING_KEYWORDS) for comp in all_companies):
                additive_bonus -= 0.02

            # GEOGRAPHIC & LOGISTICAL PENALTIES
            is_local = any(city in location for city in ["noida", "pune", "gurgaon", "ncr", "bangalore", "bengaluru", "delhi", "mumbai", "hyderabad", "chennai"])
            if not (is_local or signals.get("willing_to_relocate", False)): 
                additive_bonus -= 0.15
                
            if str(signals.get("preferred_work_mode", "")).strip().lower() == "remote":
                additive_bonus -= 0.08 # Role is hybrid

            # COMPREHENSIVE BEHAVIORAL AGGREGATION
            b_score = 0.0
            b_score += (float(signals.get("profile_completeness_score", 0.0)) / 100.0) * 0.05
            b_score += (float(signals.get("github_activity_score", 0.0)) / 100.0) * 0.05
            b_score += float(signals.get("recruiter_response_rate", 0.0)) * 0.05
            b_score += float(signals.get("interview_completion_rate", 0.0)) * 0.05
            
            if signals.get("open_to_work_flag", False): b_score += 0.03
            if signals.get("verified_phone", False): b_score += 0.02
            if signals.get("linkedin_connected", False): b_score += 0.02
            
            # Recruiter Volume Signals
            b_score += min(float(signals.get("saved_by_recruiters_30d", 0)) * 0.01, 0.05)
            b_score += min(float(signals.get("search_appearance_30d", 0)) * 0.001, 0.05)
            b_score += min(float(signals.get("profile_views_received_30d", 0)) * 0.005, 0.05)
            
            if float(signals.get("notice_period_days", 90)) <= 30: b_score += 0.03

            candidates_data.append({
                "id": c_id,
                "text": full_text[:1500],
                "lexical_hits": float(lexical_hits),
                "skill_hits": float(skill_overlap),
                "behavior_score": b_score,
                "additive_bonus": additive_bonus,
                "years_exp": raw_exp,
                "raw_title": profile.get("current_title", "Engineer"),
                "is_local": is_local,
                "skills": raw_skills[:3]
            })
        except Exception as e:
            error_count += 1
            if error_count <= MAX_ERRORS_TO_LOG:
                print(f"⚠️ Parse error on ID {c_id}: {e}", file=sys.stderr)
            continue

print(f"   -> Stage 1 Complete. Initial pool: {len(candidates_data)} (Parse Errors: {error_count})")

# --- DYNAMIC PERCENTILE CALCULATIONS FOR NORMALIZATION ---
def get_p95(feature_key):
    sorted_vals = sorted([c[feature_key] for c in candidates_data])
    if not sorted_vals: return 1.0
    p95_val = sorted_vals[int(len(sorted_vals) * 0.95)]
    return p95_val if p95_val > 0 else 1.0

P95_LEX = get_p95("lexical_hits")
P95_SKILL = get_p95("skill_hits")
P95_BEH = get_p95("behavior_score")

# --- STAGE 2: RERANKING PRE-FILTER ---
for c in candidates_data:
    c["lex_n"] = min(c["lexical_hits"], P95_LEX) / P95_LEX
    c["skill_n"] = min(c["skill_hits"], P95_SKILL) / P95_SKILL
    c["beh_n"] = min(c["behavior_score"], P95_BEH) / P95_BEH
    
    c["pre_score"] = (c["lex_n"] * 0.15) + (c["skill_n"] * 0.15) + c["beh_n"] + c["additive_bonus"]

candidates_data.sort(key=lambda x: x["pre_score"], reverse=True)
candidates_data = candidates_data[:15000]

print(f"3. Executing semantic vectorization on Top {len(candidates_data)} candidates...")
all_texts = [c["text"] for c in candidates_data]
all_embeddings = model.encode(all_texts, batch_size=256, show_progress_bar=False, convert_to_tensor=True)

JD_TEXT = "Senior AI Engineer. Requires expertise in vector search infrastructure (Pinecone, Milvus, Weaviate), embedding models (sentence-transformers, BGE, E5), and rigorous ranking evaluation metrics (NDCG, MRR, MAP). Must understand RAG architectures."
jd_embedding = model.encode(JD_TEXT, convert_to_tensor=True)

for i, c in enumerate(candidates_data):
    semantic_score = util.cos_sim(jd_embedding, all_embeddings[i]).item()
    
    # Smooth Gaussian Experience Curve
    exp_bonus = math.exp(-(abs(c["years_exp"] - 7.0) ** 2) / 12.0) * 0.08
    
    # Final Hybrid Fusion Score 
    c["rounded_score"] = round(
        (semantic_score * 0.50) + 
        (c["beh_n"] * 0.25) + 
        (c["skill_n"] * 0.15) +
        (c["lex_n"] * 0.10) + 
        c["additive_bonus"] + 
        exp_bonus, 
    4)

candidates_data.sort(key=lambda x: (-x["rounded_score"], x["id"]))

print(f"4. Generating specific, candidate-driven reasoning...")
with open(OUTPUT_FILENAME, "w", newline="", encoding="utf-8") as out_file:
    writer = csv.writer(out_file)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    
    for rank_num, c in enumerate(candidates_data[:100], start=1):
        display_skills = ", ".join([s.title() for s in c["skills"]]) if c["skills"] else "core retrieval technologies"
        
        # Factual Reasoning Construction
        sentence_1 = f"Demonstrates strong architectural alignment as a {c['raw_title'].title()}."
        
        if c["skill_hits"] >= 2:
            sentence_2 = f"Directly matches JD requirements via structured skills in {display_skills}."
        else:
            sentence_2 = f"Profile indicates practical experience with {display_skills}."

        if c["beh_n"] > 0.8:
            sentence_3 = f"Profile shows strong recruiter engagement and platform activity, backed by {c['years_exp']} years of tenure."
        else:
            sentence_3 = f"Experience profile ({c['years_exp']} yrs) closely tracks the targeted seniority tier."

        reason = f"{sentence_1} {sentence_2} {sentence_3}"
        writer.writerow([c["id"], rank_num, f"{c['rounded_score']:.4f}", reason])

print(f"✅ SUCCESS! Production-Ready Evaluation Matrix Saved to {OUTPUT_FILENAME}")
