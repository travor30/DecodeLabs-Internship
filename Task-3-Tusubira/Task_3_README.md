#  Task 3 — AI Recommendation Logic

**DecodeLabs Industrial Training | Batch 2026 | Artificial Intelligence**



##  Project Overview

A **Tech Stack Recommender** — a content-based filtering engine that maps a user's skills to the most relevant job roles using **TF-IDF vectorization** and **Cosine Similarity**. The user enters 3+ skills, the engine converts them and 15 job roles into mathematical vectors in a shared vocabulary space, then ranks all roles by angular similarity. This is the same fundamental logic powering recommendation engines at Netflix, Amazon, and LinkedIn.

> *"We are now building systems to cure 'Choice Overload.' Recommendation engines serve as digital matchmakers, connecting users to their specific needs before those needs are explicitly articulated."*
> — DecodeLabs Project Brief



##  Objectives
Implement **Content-Based Filtering** from scratch using only mathematical similarity
Understand and apply **TF-IDF weighting** to penalize generic terms and reward specific ones
Use **Cosine Similarity** (angle-based) instead of Euclidean distance (magnitude-sensitive)
Build the complete **4-step ranking pipeline**: Ingestion → Scoring → Sorting → Filtering
Handle the **Cold Start Problem** with a trending fallback mechanism



##  Verified Results

 Input Skills  #1 Match  Score

python, machine learning, tensorflow, deep learning, statistics | Data Scientist | 48.2% |
 aws, docker, kubernetes, linux, ci cd, terraform | DevOps Engineer | 50.6% |
 html, css, javascript, react, nodejs, sql | Full Stack Developer | 47.3% |



##  System Architecture


INPUT (User State)         PROCESS (Similarity Engine)        OUTPUT (Top-N List)

User enters 3+ skills     TF-IDF vectorize all docs        Rank 1: DevOps Eng.  50.6%
                           (15 roles + user profile)          Rank 2: Cloud Arch.  15.0%
                                                            Rank 3: Backend Dev.  7.8%
                           Cosine similarity scores           
                           (user vector vs each role)
                           
                          Sort descending → Top-5




##  Knowledge Base

15 job roles covering the full tech landscape:

 Category  Roles 
Data / AI | Data Scientist, AI Engineer, Data Engineer, Machine Learning Engineer 
Development | Backend Developer, Frontend Developer, Full Stack Developer, Mobile Developer
Infrastructure | DevOps Engineer, Cloud Architect, Systems Administrator 
Security & Specialized | Cybersecurity Analyst, Database Administrator, Blockchain Developer, Game Developer 

**Vocabulary size:** 402 unique terms (single words + two-word phrases via `ngram_range=(1,2)`)



##  Key Concepts Demonstrated

### Why TF-IDF Over Simple Counting

Simple word overlap treats "software" and "kubernetes" the same. TF-IDF solves this:

Component Formula Effect 

**TF** (Term Frequency) | count(term) / total_terms | Terms frequent in THIS document score higher 
**IDF** (Inverse Doc Freq) | log(total_docs / docs_with_term) | Terms common ACROSS all docs score lower 
**TF-IDF** | TF × IDF | Specific terms (kubernetes) score HIGH. Generic terms (software) score LOW

### Why Cosine Similarity Over Euclidean Distance

 **Euclidean** measures physical distance — sensitive to vector magnitude. A role with 20 tags has a larger vector than one with 5, unfairly inflating distance.
 **Cosine** measures the **angle** between vectors — magnitude-invariant. Only the direction (proportional content) matters.


cosine_similarity(A, B) = (A · B) / (|A| × |B|)

Score 1.0 = Identical direction (perfect match)
Score 0.0 = Orthogonal (no shared features)


### The 4-Step Ranking Pipeline


Step 1: Ingestion  → Collect 3+ user skills, join into document string
Step 2: Scoring    → cosine_similarity(user_vector, role_matrix)
Step 3: Sorting    → sorted(..., key=score, reverse=True)
Step 4: Filtering  → results[:top_n]  — prevents choice overload


### Cold Start Problem & Solution
A new user with no skills = zero vector → all similarity scores = 0. **Solution:** trending fallback list shows globally popular roles until enough input is collected.



##  File Structure
Task-3-AI-Recommendation/
 recommender.py       Full recommendation engine (run this)
README.md            This file




##  How to Run

**Requirements:** Python 3.6+

bash
# Install dependencies
pip install scikit-learn pandas numpy

# Run the recommender
python recommender.py


**Two modes available:**
[1]` Live mode — enter your own skills interactively
[2]` Demo mode — runs 3 preset profiles automatically

**Live Mode Example:**

  Skill 1: python
  Skill 2: machine learning
  Skill 3: tensorflow
  (press Enter to get recommendations)

  🥇 Data Scientist       Match: 48.2%  [█████████░░░░░░░░░░░]
  🥈 AI Engineer          Match: 40.8%  [████████░░░░░░░░░░░░]
  🥉 Machine Learning Eng Match: 29.1%  [█████░░░░░░░░░░░░░░░]




##  Skills Demonstrated

`Python` `scikit-learn` `TF-IDF` `Cosine Similarity` `Content-Based Filtering` `NLP Feature Extraction` `Recommendation Systems` `Vector Mathematics` `IPO Architecture`



