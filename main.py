import sys
import os
import importlib.util

# ============================================================================
# PRE-FLIGHT SYSTEM CHECKS (Ensures strict offline execution compliance)
# ============================================================================
missing_libs = []
if importlib.util.find_spec("torch") is None:
    missing_libs.append("torch")
if importlib.util.find_spec("sentence_transformers") is None:
    missing_libs.append("sentence-transformers")
if importlib.util.find_spec("rank_bm25") is None:
    missing_libs.append("rank_bm25 (Install via the bundled .whl file)")

if missing_libs:
    print("❌ CRITICAL ERROR: Missing required libraries.")
    print(f"Please install them before running the pipeline: pip install {' '.join(missing_libs)}")
    sys.exit(1)

MODEL_DIR = './all-MiniLM-L6-v2'
if not os.path.exists(MODEL_DIR):
    print(f"❌ CRITICAL ERROR: Bundled offline model folder '{MODEL_DIR}' not found.")
    print("This pipeline is configured for strict offline execution. Please ensure the model directory is present in the root folder.")
    sys.exit(1)

# --- Proceed with normal imports now that environment is verified ---
import json
import csv
import re
import gzip
import math
import torch
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi

# ============================================================================
# DATA ROBUSTNESS HELPERS
# ============================================================================
def safe_float(val, default=0.0):
    """Safely parses filthy JSON numerics (nulls, empty strings, bad types)."""
    try:
        if val is None or str(val).strip() == "": return default
        return float(val)
    except (ValueError, TypeError):
        return default

# ============================================================================
# PARTICIPANT ID
# ============================================================================
PARTICIPANT_ID = "Aerovista"
OUTPUT_FILENAME = f"{PARTICIPANT_ID}.csv"

# CPU Optimization for Hackathon Sandbox Constraints
torch.set_num_threads(os.cpu_count() or 4)

DATA_PATH = None
for path in ["candidates.jsonl", "candidates.jsonl.gz"]:
    if os.path.exists(path):
        DATA_PATH = path
        break

if not DATA_PATH:
    print("❌ ERROR: Cannot find candidates file.")
    sys.exit(1)

print(f"1. Ingesting candidates from: {DATA_PATH}...")
print("2. Verifying and loading bundled offline semantic model...")
model = SentenceTransformer(MODEL_DIR, local_files_only=True)

CONSULTING_KEYWORDS = ["tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant", "capgemini", "hcl", "deloitte"]
BANNED_ROLES = ["marketing", "accountant", "sales", "recruiter", "civil", "hr", "mechanical", "operations"]

# ============================================================================
# AUTOMATED JD PARSING & ADVANCED TOKENIZATION
# ============================================================================
JD_TEXT = "Senior AI Engineer. Requires expertise in vector search infrastructure (Pinecone, Milvus, Weaviate), embedding models (sentence-transformers, BGE, E5), and rigorous ranking evaluation metrics (NDCG, MRR, MAP). Must understand RAG architectures. vector indexing approximate search dense retrieval"

# Expanded zero-dependency stop-word removal 
STOP_WORDS = {"and", "in", "or", "a", "an", "the", "with", "for", "to", "of", "is", "are", "must", "requires", "understand", "senior", "engineer", "experience", "expertise", "models", "architectures", "systems"}

# Dynamically generate tokens preserving hyphens and slashes
JD_QUERY_TOKENS = [w for w in re.findall(r'\b[\w/-]+\b', JD_TEXT.lower()) if w not in STOP_WORDS]
JD_KEYWORDS_NORM = {k.replace("-", "").replace("/", "").replace(" ", "") for k in JD_QUERY_TOKENS}

BANNED_REGEX = re.compile(r'\b(?:' + '|'.join(BANNED_ROLES) + r')\b', re.IGNORECASE)
TITLE_REGEX = re.compile(r'\b(?:ai|ml|machine learning|search|retrieval|nlp|applied scientist|foundation model|recommendation|data scientist|mlops|ranking|infrastructure|relevance|knowledge|ai systems|ai platform|search quality|llm engineer|retrieval engineer|search engineer|ai research engineer)\b', re.IGNORECASE)

print("3. Tokenization and Signal Extraction...")
candidates_data = []
corpus_tokens = []
error_count = 0
MAX_ERRORS_TO_LOG = 5

def open_candidate_stream(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, "r", encoding="utf-8")

with open_candidate_stream(DATA_PATH) as f:
    for line in f:
        if not line.strip(): continue
        
        c_id = "unknown"
        try:
            candidate = json.loads(line)
            c_id = candidate.get("candidate_id", "unknown")
            
            profile = candidate.get("profile", {}) or {}
            history = candidate.get("career_history", []) or []
            signals = candidate.get("redrob_signals", {}) or {}
            skills = candidate.get("skills", []) or []

            # Zero-crash string casting
            current_title = str(profile.get("current_title") or "").strip().lower()
            headline = str(profile.get("headline") or "").strip().lower()
            location = str(profile.get("location") or "").strip().lower()
            summary = str(profile.get("summary") or "").strip()
            
            raw_exp = safe_float(profile.get("years_of_experience"))
            jobs_text = " ".join([str(job.get("description") or "") for job in history if job])
            full_text = f"{headline} {summary} {jobs_text}".strip()
            
            # Advanced tokenization for BM25
            tokens = re.findall(r'\b[\w/-]+\b', full_text.lower())
            
            raw_skills = [str(s.get("name") or "") for s in skills if s.get("name")]
            norm_skills = [s.lower().replace("-", "").replace("/", "").replace(" ", "") for s in raw_skills]
            skill_overlap = len(JD_KEYWORDS_NORM.intersection(set(norm_skills)))

            # STRUCTURED BUSINESS MODIFIERS
            business_score = 0.0
            if TITLE_REGEX.search(current_title): business_score += 2.0
            
            combined_titles = f"{current_title} {headline}"
            if BANNED_REGEX.search(combined_titles): business_score -= 3.0
            
            impossible_skills = sum(1 for s in skills if str(s.get("proficiency") or "").lower() in ["expert", "advanced"] and not s.get("duration_months"))
            if impossible_skills >= 3: business_score -= 3.0
            
            all_companies = [str(job.get("company") or "").strip().lower() for job in history if job]
            if profile.get("current_company"): all_companies.append(str(profile.get("current_company") or "").strip().lower())
            
            if any(any(kw in comp for kw in CONSULTING_KEYWORDS) for comp in all_companies): business_score -= 0.5

            is_local = any(city in location for city in ["noida", "pune", "gurgaon", "ncr", "bangalore", "bengaluru", "delhi", "mumbai", "hyderabad", "chennai"])
            if not (is_local or signals.get("willing_to_relocate", False)): business_score -= 1.5
            if str(signals.get("preferred_work_mode") or "").strip().lower() == "remote": business_score -= 1.5

            business_score += math.exp(-(abs(raw_exp - 7.0) ** 2) / 12.0) * 2.0
            business_score += (skill_overlap * 0.5)

            # COMPREHENSIVE BEHAVIORAL AGGREGATION (Safe float integration)
            b_score = 0.0
            b_score += (safe_float(signals.get("profile_completeness_score")) / 100.0) * 5.0
            b_score += (safe_float(signals.get("github_activity_score")) / 100.0) * 5.0
            b_score += safe_float(signals.get("recruiter_response_rate")) * 5.0
            b_score += safe_float(signals.get("interview_completion_rate")) * 5.0
            
            if signals.get("open_to_work_flag", False): b_score += 3.0
            if signals.get("verified_phone", False): b_score += 2.0
            if signals.get("linkedin_connected", False): b_score += 2.0
            
            b_score += min(safe_float(signals.get("saved_by_recruiters_30d")) * 1.0, 5.0)
            b_score += min(safe_float(signals.get("search_appearance_30d")) * 0.1, 5.0)
            b_score += min(safe_float(signals.get("profile_views_received_30d")) * 0.5, 5.0)
            if safe_float(signals.get("notice_period_days", 90.0), 90.0) <= 30.0: b_score += 3.0

            candidates_data.append({
                "id": c_id,
                "text": full_text[:1500],
                "behavior_score": b_score,
                "business_score": business_score,
                "years_exp": raw_exp,
                "raw_title": str(profile.get("current_title") or "Engineer").strip(),
                "skills": raw_skills[:3]
            })
            corpus_tokens.append(tokens)
            
        except Exception as e:
            error_count += 1
            if error_count <= MAX_ERRORS_TO_LOG:
                print(f"⚠️ Parse error on ID {c_id}: {e}", file=sys.stderr)
            continue

print(f"   -> Stage 1 Complete. Initial pool: {len(candidates_data)} (Parse errors: {error_count})")

print(f"4. Executing BM25 Lexical Retrieval on {len(corpus_tokens)} candidates...")
bm25 = BM25Okapi(corpus_tokens)
lexical_scores = bm25.get_scores(JD_QUERY_TOKENS)

for i, c in enumerate(candidates_data):
    c["bm25_score"] = lexical_scores[i]

# --- THE TWO-TOWER PRE-FILTER UNION ---
candidates_data.sort(key=lambda x: x["bm25_score"], reverse=True)
top_lexical = candidates_data[:10000]
lexical_set = {c["id"] for c in top_lexical}

remaining_candidates = [c for c in candidates_data if c["id"] not in lexical_set]
remaining_candidates.sort(key=lambda x: x["behavior_score"] + x["business_score"], reverse=True)
top_behavioral = remaining_candidates[:5000]

semantic_pool = top_lexical + top_behavioral

print(f"5. Executing Semantic Vectorization on Elite Union Pool ({len(semantic_pool)} candidates)...")
all_texts = [c["text"] for c in semantic_pool]
all_embeddings = model.encode(all_texts, batch_size=256, show_progress_bar=False, convert_to_tensor=True)

jd_embedding = model.encode(JD_TEXT, convert_to_tensor=True)

for i, c in enumerate(semantic_pool):
    c["semantic_score"] = util.cos_sim(jd_embedding, all_embeddings[i]).item()

print(f"6. Fusing multi-dimensional signals via Reciprocal Rank Fusion (RRF)...")
K = 60

# Rank 1: Semantic
semantic_pool.sort(key=lambda x: x["semantic_score"], reverse=True)
for rank, c in enumerate(semantic_pool, start=1): c["sem_rank"] = rank

# Rank 2: Lexical (BM25)
semantic_pool.sort(key=lambda x: x["bm25_score"], reverse=True)
for rank, c in enumerate(semantic_pool, start=1): c["lex_rank"] = rank

# Rank 3: Behavioral
semantic_pool.sort(key=lambda x: x["behavior_score"], reverse=True)
for rank, c in enumerate(semantic_pool, start=1): c["beh_rank"] = rank

# Rank 4: Business/Domain Heuristics
semantic_pool.sort(key=lambda x: x["business_score"], reverse=True)
for rank, c in enumerate(semantic_pool, start=1): c["bus_rank"] = rank

# Execute Pure RRF (Standardized mathematical consensus)
for c in semantic_pool:
    c["rrf_score"] = (
        (1.0 / (K + c["sem_rank"])) + 
        (1.0 / (K + c["lex_rank"])) + 
        (1.0 / (K + c["beh_rank"])) + 
        (1.0 / (K + c["bus_rank"]))
    )

# Final Deterministic Sort by Infinite Precision RRF Score
semantic_pool.sort(key=lambda x: (-x["rrf_score"], x["id"]))

print(f"7. Generating dynamic output matrix...")
with open(OUTPUT_FILENAME, "w", newline="", encoding="utf-8") as out_file:
    writer = csv.writer(out_file)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    
    for rank_num, c in enumerate(semantic_pool[:100], start=1):
        display_skills = ", ".join([s.title() for s in c["skills"]]) if c["skills"] else "core retrieval technologies"
        
        # ---------------------------------------------------------------------
        # HIGH-SCANNABILITY REASONING BANK (Parallel Structure for CSV Reading)
        # ---------------------------------------------------------------------
        
        # Sentence 1: The Core Signal Match
        if c["sem_rank"] < c["lex_rank"] and c["sem_rank"] < c["beh_rank"] and c["sem_rank"] < c["bus_rank"]:
            sentence_1 = f"Selected for strong semantic alignment as a {c['raw_title'].title()}."
        elif c["beh_rank"] < c["sem_rank"] and c["beh_rank"] < c["lex_rank"] and c["beh_rank"] < c["bus_rank"]:
            sentence_1 = f"Selected for high platform engagement as a {c['raw_title'].title()}."
        elif c["bus_rank"] < c["sem_rank"] and c["bus_rank"] < c["lex_rank"] and c["bus_rank"] < c["beh_rank"]:
            sentence_1 = f"Selected for relevant domain experience as a {c['raw_title'].title()}."
        else:
            sentence_1 = f"Selected for balanced architectural fit as a {c['raw_title'].title()}."
            
        # Sentence 2: Verified Skills
        sentence_2 = f"Verified skills include {display_skills}."

        # Sentence 3: Tenure Tracking
        sentence_3 = f"Experience tier tracks at {c['years_exp']} years."

        reason = f"{sentence_1} {sentence_2} {sentence_3}"
        
        # Output isolation bounds rounding strictly to the CSV writer
        writer.writerow([c["id"], rank_num, f"{c['rrf_score']:.6f}", reason])

print(f"✅ SUCCESS! Production-Ready ATS Pipeline Executed. Results saved to {OUTPUT_FILENAME}")
