# AI Recommendation Logic - Tech Stack Recommender
# Project 3 - DecodeLabs Industrial Training | Batch 2026
# Topic: Content-Based Filtering | TF-IDF + Cosine Similarity

# -------------------------------------------------------
# STEP 0: IMPORTS
#
# sklearn.feature_extraction.text.TfidfVectorizer
#   → Converts text (skill tags) into weighted TF-IDF vectors
#     in a shared vocabulary space
#
# sklearn.metrics.pairwise.cosine_similarity
#   → Measures the angular similarity between two vectors.
#     Score 1 = identical orientation, Score 0 = no overlap
#
# numpy  → array operations and argsorting
# pandas → clean tabular display of results
# -------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------------------------------
# STEP 1: THE KNOWLEDGE BASE
# This is the item catalogue — the "items" in our
# recommendation engine. Each job role is described
# by a string of skill tags (its feature document).
#
# Design rule from the slides:
#   Item features and user features MUST use the exact
#   same vocabulary. Mismatches cause the similarity
#   math to silently fail (e.g. "Web Design" vs
#   "Frontend Development" would score zero overlap).
#
# Each role string = a mini "document" of skills.
# The TF-IDF vectorizer will treat these as documents.
# -------------------------------------------------------
JOB_ROLES = {
    "Data Scientist": (
        "python sql machine learning statistics data analysis "
        "pandas numpy scikit-learn visualization regression classification "
        "deep learning tensorflow pytorch jupyter"
    ),
    "AI Engineer": (
        "python machine learning deep learning tensorflow pytorch "
        "neural networks nlp computer vision transformers huggingface "
        "model deployment apis mlops scikit-learn"
    ),
    "Data Engineer": (
        "python sql spark hadoop etl pipeline postgresql mongodb "
        "kafka airflow data warehousing bigquery snowflake "
        "bash linux cloud aws data modeling"
    ),
    "Backend Developer": (
        "python java nodejs sql rest apis postgresql mongodb "
        "express django fastapi git authentication microservices "
        "docker linux aws databases"
    ),
    "Frontend Developer": (
        "html css javascript react vuejs typescript figma "
        "ui ux web design tailwind next.js responsive "
        "git npm webpack animations accessibility"
    ),
    "Full Stack Developer": (
        "html css javascript react nodejs python sql rest apis "
        "git docker postgresql mongodb express authentication "
        "cloud aws deployment responsive design"
    ),
    "DevOps Engineer": (
        "aws docker kubernetes ci cd linux bash git automation "
        "cloud infrastructure terraform ansible jenkins monitoring "
        "networking security scripting pipelines"
    ),
    "Cloud Architect": (
        "aws azure google cloud kubernetes docker terraform "
        "infrastructure networking security iam load balancing "
        "cost optimization microservices serverless monitoring"
    ),
    "Cybersecurity Analyst": (
        "networking security linux firewalls penetration testing "
        "ethical hacking python forensics cryptography siem "
        "vulnerability assessment incident response compliance"
    ),
    "Mobile Developer": (
        "java kotlin swift ios android react native flutter "
        "mobile ui apis firebase sqlite push notifications "
        "app store git testing agile"
    ),
    "Machine Learning Engineer": (
        "python machine learning scikit-learn tensorflow pytorch "
        "feature engineering model training deployment mlops "
        "docker apis sql statistics a/b testing pipelines"
    ),
    "Database Administrator": (
        "sql postgresql mysql mongodb oracle nosql "
        "database design indexing optimization backup "
        "replication linux scripting bash aws rds"
    ),
    "Systems Administrator": (
        "linux windows bash powershell networking dns dhcp "
        "virtualization vmware active directory monitoring "
        "backup security automation scripting troubleshooting"
    ),
    "Blockchain Developer": (
        "solidity ethereum blockchain smart contracts web3 "
        "javascript python cryptography consensus protocols "
        "defi nft security auditing"
    ),
    "Game Developer": (
        "c++ c# unity unreal engine game design physics "
        "graphics opengl shaders python scripting 3d "
        "animation multiplayer networking"
    ),
}

# -------------------------------------------------------
# TRENDING FALLBACK (Cold Start bypass)
# If user provides no skills, show globally popular roles.
# This solves the "User Cold Start Problem" mentioned
# in the slides — a zero-vector produces zero scores.
# -------------------------------------------------------
TRENDING_ROLES = [
    "Full Stack Developer",
    "AI Engineer",
    "Data Scientist",
    "DevOps Engineer",
    "Cloud Architect",
]


# -------------------------------------------------------
# BUILD TF-IDF MATRIX
# Creates the vectorizer and fits it on ALL documents
# (all job role strings) to learn the shared vocabulary.
#
# Returns the fitted vectorizer and the role matrix so
# we can reuse them for every user query.
# -------------------------------------------------------
def build_tfidf_matrix(job_roles):
    role_names = list(job_roles.keys())
    role_docs  = list(job_roles.values())

    # TfidfVectorizer parameters:
    # ngram_range=(1,2) → captures both single words ("python")
    #   AND two-word phrases ("machine learning")
    # min_df=1          → include all terms (small dataset)
    # sublinear_tf=True → apply log normalization to TF
    #   so a term appearing 100x isn't 100x more important than once
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True
    )

    # fit_transform on the role documents:
    # fit()      → learns the vocabulary and IDF weights
    # transform() → converts each document to a TF-IDF vector
    role_matrix = vectorizer.fit_transform(role_docs)

    return vectorizer, role_matrix, role_names


# -------------------------------------------------------
# BUILD USER PROFILE VECTOR
# Takes the user's skill list and converts it to a
# TF-IDF vector using the SAME vocabulary (vectorizer)
# that was fitted on the job roles.
#
# CRITICAL: We use .transform() NOT .fit_transform()
# This is the same principle as the scaler in Project 2 —
# fit on the "training data" (job roles), transform the
# "new input" (user skills) with the same mapping.
# -------------------------------------------------------
def build_user_vector(vectorizer, user_skills_list):
    # Join the list into a single string document
    user_doc = " ".join(user_skills_list)

    # Transform using the fitted vectorizer
    # .transform() uses the vocabulary learned from job roles
    user_vector = vectorizer.transform([user_doc])

    return user_vector


# -------------------------------------------------------
# THE 4-STEP RANKING PIPELINE
# As described in the slides:
#   Step 1 — Ingestion  : user skills already collected
#   Step 2 — Scoring    : calculate cosine similarity for each role
#   Step 3 — Sorting    : sort scores descending
#   Step 4 — Filtering  : return only Top-N results
# -------------------------------------------------------
def rank_recommendations(user_vector, role_matrix, role_names, top_n=5):

    # --- STEP 2: SCORING ---
    # cosine_similarity returns a 2D array of shape (1, num_roles)
    # We take [0] to get the flat array of scores
    similarity_scores = cosine_similarity(user_vector, role_matrix)[0]

    # Pair each role name with its score
    scored_roles = list(zip(role_names, similarity_scores))

    # --- STEP 3: SORTING ---
    # Sort by score descending — highest match first
    sorted_roles = sorted(scored_roles, key=lambda x: x[1], reverse=True)

    # --- STEP 4: FILTERING (Top-N) ---
    top_results = sorted_roles[:top_n]

    return top_results


# -------------------------------------------------------
# COLLECT USER INPUT
# Minimum 3 skills required (per project specification).
# Input is sanitized: lowercased, stripped, deduplicated.
# -------------------------------------------------------
def collect_user_input():
    print()
    print("  Enter your skills one by one.")
    print("  Minimum 3 skills required for accurate matching.")
    print("  Type 'done' when finished (or press Enter on empty line).")
    print()

    skills = []
    while True:
        skill = input(f"  Skill {len(skills) + 1}: ").strip().lower()

        # Exit conditions
        if skill in ("done", "exit", "quit", "") and len(skills) >= 3:
            break
        elif skill in ("done", "exit", "quit", "") and len(skills) < 3:
            print(f"  ⚠  Please enter at least 3 skills. You have {len(skills)} so far.")
            continue

        # Skip empty input mid-stream
        if not skill:
            continue

        # Avoid exact duplicates
        if skill in skills:
            print(f"  '{skill}' already added, skipping.")
            continue

        skills.append(skill)
        print(f"  ✓ Added: '{skill}'")

        # After 3+ skills, let user know they can stop
        if len(skills) == 3:
            print()
            print("  You have 3 skills. Keep adding or press Enter to get recommendations.")
            print()

    return skills


# -------------------------------------------------------
# DISPLAY RESULTS
# Formats and prints the Top-N recommendations with
# score bars, role descriptions, and key skills.
# -------------------------------------------------------
def display_results(top_results, user_skills):
    print()
    print("=" * 60)
    print("  RECOMMENDATION RESULTS")
    print("=" * 60)
    print(f"  Your Skills   : {', '.join(user_skills)}")
    print(f"  Method        : TF-IDF Vectorization + Cosine Similarity")
    print(f"  Top {len(top_results)} Matches Found")
    print("=" * 60)
    print()

    # Role descriptions for display
    role_descriptions = {
        "Data Scientist":           "Analyze data, build predictive models, extract insights",
        "AI Engineer":              "Build and deploy ML/DL models and AI-powered systems",
        "Data Engineer":            "Design data pipelines, warehouses, and ETL processes",
        "Backend Developer":        "Build server-side APIs, databases, and core logic",
        "Frontend Developer":       "Build user interfaces and web experiences",
        "Full Stack Developer":     "Handle both frontend UI and backend server logic",
        "DevOps Engineer":          "Automate infrastructure, CI/CD, and cloud deployments",
        "Cloud Architect":          "Design scalable, secure cloud infrastructure",
        "Cybersecurity Analyst":    "Protect systems, detect threats, and ensure compliance",
        "Mobile Developer":         "Build iOS and Android applications",
        "Machine Learning Engineer":"Train, optimize, and deploy ML models at scale",
        "Database Administrator":   "Manage and optimize databases and data storage",
        "Systems Administrator":    "Maintain servers, networks, and IT infrastructure",
        "Blockchain Developer":     "Build decentralized apps and smart contracts",
        "Game Developer":           "Design and code interactive game experiences",
    }

    for rank, (role, score) in enumerate(top_results, 1):
        # Convert score to percentage
        pct = score * 100

        # Visual score bar (out of 20 blocks)
        filled  = int(pct / 5)
        bar     = "█" * filled + "░" * (20 - filled)

        # Medal for top 3
        medal = ["🥇", "🥈", "🥉", "  4.", "  5."][rank - 1]

        print(f"  {medal}  {role}")
        print(f"       Match Score : {pct:.1f}%")
        print(f"       [{bar}]")
        desc = role_descriptions.get(role, "")
        if desc:
            print(f"       {desc}")
        print()

    print("=" * 60)
    print()


# -------------------------------------------------------
# DEMO MODE
# Runs 3 preset user profiles without manual input.
# Useful for showing the system working in a video.
# -------------------------------------------------------
def run_demo(vectorizer, role_matrix, role_names):
    demo_profiles = [
        {
            "name"  : "Profile A — ML / Data focus",
            "skills": ["python", "machine learning", "tensorflow", "deep learning", "statistics"]
        },
        {
            "name"  : "Profile B — Cloud / Infrastructure focus",
            "skills": ["aws", "docker", "kubernetes", "linux", "ci cd", "terraform"]
        },
        {
            "name"  : "Profile C — Web development focus",
            "skills": ["html", "css", "javascript", "react", "nodejs", "sql"]
        },
    ]

    print()
    print("=" * 60)
    print("  DEMO MODE — Running 3 Sample Profiles")
    print("=" * 60)

    for profile in demo_profiles:
        print()
        print(f"  ── {profile['name']} ──")
        print(f"  Input Skills : {', '.join(profile['skills'])}")
        print()

        user_vector = build_user_vector(vectorizer, profile["skills"])
        results     = rank_recommendations(user_vector, role_matrix, role_names, top_n=3)

        print(f"  {'Rank':<5}  {'Job Role':<28}  {'Score':>8}")
        print("  " + "-" * 46)
        for rank, (role, score) in enumerate(results, 1):
            bar = "█" * int(score * 20)
            print(f"  {rank:<5}  {role:<28}  {score*100:>6.1f}%  {bar}")
        print()

    print("=" * 60)


# -------------------------------------------------------
# SHOW SCORE TABLE (bonus)
# Shows ALL job roles ranked by similarity to user.
# Great for the video — demonstrates the full scoring.
# -------------------------------------------------------
def show_full_score_table(top_results, role_names, all_scores):
    print()
    print("  Full Scoring Table (All Roles):")
    print(f"  {'Role':<30}  {'Score':>8}  {'Bar'}")
    print("  " + "-" * 55)

    all_scored = sorted(zip(role_names, all_scores), key=lambda x: x[1], reverse=True)
    for role, score in all_scored:
        bar = "█" * int(score * 20)
        marker = " ← TOP MATCH" if role == all_scored[0][0] else ""
        print(f"  {role:<30}  {score*100:>7.1f}%  {bar}{marker}")
    print()


# -------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------
def main():
    print()
    print("=" * 60)
    print("   PROJECT 3: AI Recommendation Logic")
    print("   Tech Stack Recommender")
    print("   Content-Based Filtering | TF-IDF + Cosine Similarity")
    print("=" * 60)

    # --- BUILD THE ENGINE ONCE ---
    print()
    print("  [INIT] Building TF-IDF knowledge base...")
    vectorizer, role_matrix, role_names = build_tfidf_matrix(JOB_ROLES)
    print(f"  [INIT] Vocabulary size : {len(vectorizer.vocabulary_)} unique terms")
    print(f"  [INIT] Job roles loaded: {len(role_names)}")
    print()

    # --- ASK USER: DEMO or LIVE ---
    print("  How would you like to run this?")
    print("  [1] Enter my own skills (live mode)")
    print("  [2] Run demo with preset profiles")
    print()
    choice = input("  Enter 1 or 2: ").strip()

    # ==================== DEMO MODE ====================
    if choice == "2":
        run_demo(vectorizer, role_matrix, role_names)
        return

    # ==================== LIVE MODE ====================
    print()
    print("  Great! Let's build your profile.")

    # --- PHASE 1: INPUT (Ingestion) ---
    user_skills = collect_user_input()

    # Cold Start check — if somehow we have no skills
    if not user_skills:
        print()
        print("  No skills entered. Showing trending roles instead:")
        for i, role in enumerate(TRENDING_ROLES, 1):
            print(f"    {i}. {role}")
        return

    # --- PHASE 2: PROCESS (Vector + Scoring + Sorting) ---
    print()
    print("  [PROCESS] Converting your skills to TF-IDF vector...")
    user_vector = build_user_vector(vectorizer, user_skills)

    print("  [PROCESS] Calculating cosine similarity against all roles...")
    # Get all scores for the full table display
    all_scores  = cosine_similarity(user_vector, role_matrix)[0]
    top_results = rank_recommendations(user_vector, role_matrix, role_names, top_n=5)

    # --- PHASE 3: OUTPUT ---
    display_results(top_results, user_skills)

    # Show full scoring table
    show_full_score_table(top_results, role_names, all_scores)

    # --- BONUS: Run Again? ---
    again = input("  Would you like to test another profile? (yes/no): ").strip().lower()
    if again in ("yes", "y"):
        main()
    else:
        print()
        print("  Thanks for using the Tech Stack Recommender!")
        print("  Your path is now mapped. 🚀")
        print("=" * 60)
        print()


# -------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------
if __name__ == "__main__":
    main()
