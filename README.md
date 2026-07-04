# 🎓 AI-Powered Placement Preparation Ecosystem

> An end-to-end AI platform that analyzes resumes, scores them against job descriptions (ATS), predicts placement probability, identifies skill gaps, generates personalized learning roadmaps, and runs a RAG-powered mock interview system — built as an internship capstone project.
*Google Colab Link*: https://colab.research.google.com/drive/16QRpSoagXyOF8fQgiylb_ywweLbSsUSM?usp=sharing

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow)
![Gemini](https://img.shields.io/badge/Gemini%203.5%20Flash-LLM-4285F4?logo=google)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-yellow)
![Gradio](https://img.shields.io/badge/Gradio-UI-fb7500?logo=gradio)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📋 Internship Details

| Field | Detail |
|---|---|
| **Internship Title** | AI/ML Internship |
| **Organization** | Coding Block Schools of Technology |
| **Duration** | June 2026 – July 2026 |
| **Mentor / Supervisor** | Aryesh Rai |
| **Internship Type** | Remote |

## 👩‍💻 Candidate Details

| Field | Detail |
|---|---|
| **Name** | Kanishka Pal |
| **Institution** | Hi-Tech Institute of Engineering and Technology Ghaziabad |
| **Course / Branch** | Computer Science and Engineering |
| **Roll No. / Registration No.** |2402200100024 |
| **Email** | kksnehapal@gmail.com|
| **LinkedIn** | https://www.linkedin.com/in/kanishkapal2005/ |

---

## 📌 Project Name

**AI-Powered Placement Preparation Ecosystem**

---

## 📖 Project Description

Placement preparation is fragmented by nature — students juggle resume polishing, ATS
compliance, aptitude/technical prep, mock interviews, and role-specific skill-building
across a dozen disconnected tools. This project unifies all of that into **one AI-powered
pipeline** built entirely in a single Google Colab notebook.

The system combines **classical Machine Learning, Deep Learning, NLP, and
Retrieval-Augmented Generation (RAG)** into ten integrated modules:

- Parses resumes (PDF) and extracts structured candidate data with **NLP** (spaCy +
  keyword/skill rules).
- Scores a resume against a job description using **TF-IDF + Sentence-Transformer
  semantic similarity** to produce an ATS score.
- Predicts placement probability from clinical-style profile data using **Logistic
  Regression, Random Forest, XGBoost**, and a **Keras ANN** (dense + BatchNorm + Dropout +
  EarlyStopping).
- Performs **skill-gap analysis** against six target job roles.
- Generates **personalized daily/weekly/monthly learning roadmaps** using the **Gemini
  3.5 Flash** LLM.
- Runs a **RAG-based interview preparation system** (Sentence-Transformer embeddings +
  FAISS vector index + Gemini 3.5 Flash) covering DSA, HR, System Design, DBMS, OS,
  Computer Networks, OOP, and company-specific interview experiences.
- Simulates a full **AI Mock Interviewer** that asks role-specific questions and scores
  answers on five rubric dimensions.
- Visualizes every result on an **Analytics Dashboard** (Plotly gauges, radar charts,
  feature-importance bars).
- Wraps all ten modules into one multi-tab **Gradio application**.

---

## ❓ Problem Statement

> **Given a student's resume, target job description, profile metrics, and desired job
> role, predict their placement readiness, identify the specific skill and resume gaps
> holding them back, and generate a personalized, actionable plan (roadmap + mock
> interview feedback) to close those gaps.**

**Why it matters:**
- Placement outcomes depend on many interacting factors (academics, DSA, communication,
  projects, certifications) — no single metric like CGPA tells the full story.
- Most students don't get structured, individualized feedback on *why* their resume or
  interview performance falls short — feedback is generic or comes too late.
- Combining predictive ML with a generative, explainable LLM layer turns a static score
  into a concrete, personalized action plan — closer to what a real placement mentor
  would give.

**ML/AI formulation:** A mixed pipeline — **binary classification** (placement
probability), **unsupervised/rule-based similarity scoring** (ATS, skill gap), and
**generative AI with retrieval grounding** (roadmap generation, RAG interview Q&A, mock
interview evaluation).

---

## 🗂️ Datasets

| Dataset | Description | Source |
|---|---|---|
| **Placement Dataset** | 2,000 synthetic student records — `CGPA, DSA score, Aptitude score, Communication score, Projects count, Internship experience, Certifications count, Hackathon participation, Soft skills score → Placed (0/1)` | Generated in-notebook (`generate_placement_dataset()`), fixed `seed=42` for reproducibility |
| **Sample Resume** | Auto-generated demo PDF resume (name, contact, education, skills, projects) so the pipeline is testable without uploading a real file | Generated in-notebook via `reportlab` |
| **Interview Knowledge Base** | Seeded Q&A corpus across DSA, HR, System Design, DBMS, OS, Computer Networks, OOP, and company interview experiences, chunked and embedded for retrieval | Curated in-notebook (`INTERVIEW_DOCS`) |
| **Role Skill Profiles** | Keyword sets mapping 7 job categories (Web Developer, Data Scientist, AI Engineer, ML Engineer, Backend Developer, Full Stack Developer, Software Engineer) and 6 target roles (for skill-gap analysis) to required skills | Curated in-notebook (`SKILL_KEYWORDS`, `ROLE_REQUIREMENTS`) |

**File paths used in the notebook:**
```
AI_Placement_Ecosystem/datasets/placement_dataset.csv
AI_Placement_Ecosystem/resumes/sample_resume.pdf
AI_Placement_Ecosystem/embeddings/interview_faiss.index
```

---

## 🛠️ Tech Stack

| Category | Tools / Libraries |
|---|---|
| Language | Python 3.10 |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn, XGBoost |
| Deep Learning | TensorFlow / Keras |
| NLP | spaCy, Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Retrieval-Augmented Generation | FAISS (vector index), Sentence-Transformers |
| LLM | **Gemini 3.5 Flash** (`gemini-3.5-flash`, via `google-generativeai`) |
| Resume Parsing | pdfplumber, regex |
| Model Persistence | Joblib (`.pkl`), Keras native (`.keras`) |
| Web Interface | Gradio (multi-tab Blocks app) |
| PDF Generation (demo data) | ReportLab |
| Environment | Google Colab |

---

## 🧭 Project Pipeline (High-Level Flow)

```
Resume PDF ──► Resume Analyzer (pdfplumber + spaCy + keyword rules) ──► Skills / Category
                                                                          │
Job Description ─────────► ATS Score Predictor (TF-IDF + SBERT) ◄────────┘
                                                                          │
Student Profile ─────────► Placement Predictor (LogReg / RF / XGBoost / Keras ANN)
                                                                          │
Skills + Target Role ────► Skill Gap Analysis ──► Learning Roadmap (Gemini 3.5 Flash)
                                                                          │
Interview Question ──────► RAG Pipeline (SBERT + FAISS) ──► Gemini 3.5 Flash ──► Answer
                                                                          │
Resume + Role ────────────► AI Mock Interviewer (Gemini 3.5 Flash) ──► Score + Feedback
                                                                          │
All Results ──────────────► Analytics Dashboard (Plotly) ──► Gradio App (10 tabs)
```

---

## 🔍 Step-by-Step Explanation of the Notebook (Module by Module)

The notebook is organized into 10 functional modules across ~33 cells. Below is what
every module's code actually does, and why.

### 0. Setup — Installs, Imports, Project Structure & Gemini Wrapper
- A single install cell pulls every dependency (`pdfplumber, spacy, sentence-transformers,
  faiss-cpu, xgboost, gradio, plotly, reportlab, google-generativeai`) plus the spaCy
  English model, and force-reinstalls `pillow` to avoid a known Colab
  `torchvision`/`Pillow` conflict (`ImportError: cannot import name 'ImageDraw'`) — the
  notebook explicitly instructs a **Runtime → Restart session** after this cell, since
  Colab needs a fresh process for the fix to take effect.
- One imports cell groups every library so dependencies are visible at a glance.
- A folder-creation loop builds the full `AI_Placement_Ecosystem/` project structure
  (`datasets/, models/, embeddings/, resumes/, notebooks/, rag/, utils/,
  visualizations/, app/, outputs/`).
- A `GeminiClient` class wraps `google.generativeai`, handling **retries with
  exponential backoff** and graceful error strings instead of raw exceptions — every
  LLM-powered module below calls through this one class, using the
  `gemini-3.5-flash` model.

### 1. Resume Analyzer
```python
SKILL_KEYWORDS = {"Web Developer": [...], "Data Scientist": [...], "AI Engineer": [...], ...}
```
- `extract_text_from_pdf()` reads a PDF with `pdfplumber`.
- `extract_contact_info()` regex-matches email/phone and takes the first non-empty line
  as the name.
- `extract_skills()` matches resume text against a master skill vocabulary spanning 7
  job categories.
- `extract_education()` / `extract_projects()` keyword-filter relevant lines (degree
  names, "project" mentions).
- `categorize_candidate()` scores keyword overlap between detected skills and each
  role's required skill set, returning the best-matching category plus per-role scores.
- `analyze_resume()` ties all of the above into one function returning contact info,
  detected skills, missing skills (for the predicted category), education, projects,
  and category scores.
- A demo resume PDF is generated on the fly with `reportlab` so the whole pipeline is
  testable immediately, without requiring a real uploaded file.

### 2. ATS Score Predictor
```python
overall_score = (0.3*keyword_similarity + 0.3*skill_match_pct + 0.4*semantic_similarity) * 100
```
- **Keyword similarity** — TF-IDF vectors of resume vs. job description, compared with
  cosine similarity.
- **Skill match %** — rule-based overlap between skills detected in the resume and
  skills detected in the job description.
- **Semantic similarity** — Sentence-Transformer embeddings of the full resume and job
  description texts, compared with cosine similarity (captures meaning, not just exact
  keyword overlap).
- The three signals are weighted into one **0–100 ATS score**, alongside a list of
  missing keywords and generated improvement suggestions.

### 3. Placement Prediction Model
- `generate_placement_dataset()` builds 2,000 synthetic but realistically-correlated
  student records: a weighted linear combination of all 9 profile features plus Gaussian
  noise determines a latent "readiness score," and the top ~55% (by percentile threshold)
  are labeled placed.
- **Train/test split** is stratified 80:20; features are standardized with
  `StandardScaler` (fit on train only).
- **Three models** are trained and benchmarked head-to-head: **Logistic Regression**
  (linear baseline), **Random Forest**, and **XGBoost**.
- Each model is scored on **Accuracy, Precision, Recall, F1, and ROC-AUC**, assembled
  into one comparison table.
- **XGBoost feature importances** are plotted (Plotly horizontal bar) to show which
  profile features drive predictions most.
- The best model + scaler are persisted with **Joblib**, and wrapped in a reusable
  `predict_placement()` function for live inference.

### 4. Skill Gap Analysis
```python
ROLE_REQUIREMENTS = {"MERN Developer": [...], "AI Engineer": [...], "Data Scientist": [...], ...}
```
- `skill_gap_analysis()` compares a student's current skill list against the required
  skill set for one of six target roles (MERN Developer, AI Engineer, Data Scientist,
  Backend Developer, Frontend Developer, Full Stack Developer), returning present
  skills, missing skills, and a **skill readiness percentage**.

### 5. Personalized Learning Roadmap Generator (Gemini 3.5 Flash)
- `generate_learning_roadmap()` sends the student's profile and skill-gap result to
  Gemini 3.5 Flash with a strict JSON-only prompt template, asking for a **daily,
  weekly, and monthly roadmap**, recommended projects, and an interview-prep strategy.
- The response is parsed as JSON (stripping any markdown fences); if parsing fails, the
  raw text is returned as a fallback so the app never hard-crashes on a malformed LLM
  response.

### 6. RAG-Based Interview Preparation System
- A seeded corpus (`INTERVIEW_DOCS`) covers DSA, HR, System Design, DBMS, OS, Computer
  Networks, OOP, and company-specific interview experiences.
- `chunk_documents()` splits each document into fixed-size word chunks.
- Chunks are embedded with **Sentence-Transformers** and indexed in a **FAISS**
  `IndexFlatL2` vector store (persisted to disk).
- `retrieve_context()` does a nearest-neighbor search for the top-k relevant chunks
  given a query.
- `rag_answer()` feeds the retrieved context + question into **Gemini 3.5 Flash**,
  grounding the LLM's answer in the retrieved material instead of letting it hallucinate
  freely, and returns both the answer and its sources.

### 7. AI Mock Interviewer
- `generate_interview_questions()` asks Gemini 3.5 Flash for *n* role- and
  experience-level-specific technical questions, returned as a JSON list (with a
  line-split fallback if JSON parsing fails).
- `evaluate_interview_answer()` sends each question/answer pair back to Gemini 3.5
  Flash, asking it to score **technical correctness, confidence, communication,
  relevance, and completeness** (0–10 each) plus a written feedback paragraph, all as
  strict JSON.
- `run_mock_interview()` orchestrates a full mock round: generate questions → collect
  answers → evaluate each → return the full scored transcript.

### 8. Deep Learning Performance Predictor
- `build_ann_model()` defines a Keras `Sequential` network: `Dense(64) → BatchNorm →
  Dropout(0.3) → Dense(32) → BatchNorm → Dropout(0.2) → Dense(16) → Dense(1, sigmoid)`.
- Trained on the same scaled placement dataset with `EarlyStopping` (`patience=10`,
  restoring best weights) to prevent overfitting.
- Training/validation **loss and accuracy curves** are plotted and saved, and a full
  `classification_report` is printed on the held-out test set — giving a direct,
  apples-to-apples comparison against the classical ML models from Module 3.

### 9. Analytics Dashboard
- Reusable Plotly helper functions turn every module's output into a visual: **ATS
  score gauge**, **placement probability gauge**, **skill readiness gauge**, and an
  **interview performance radar chart** (one axis per rubric dimension).

### 10. Gradio Application
- All nine functional modules above are wrapped in `ui_*` functions and assembled into
  one `gr.Blocks` app with **8 tabs**: Resume Analyzer, ATS Checker, Placement
  Predictor, Skill Gap Analysis, Learning Roadmap, RAG Interview Assistant, AI Mock
  Interviewer, and an Analytics Dashboard summarizing model metrics and feature
  importance.
- Launched with `share=True` for an instant public link directly from Colab.

---

## 📊 Model Performance Comparison — Placement Prediction

Computed on the synthetic dataset (`n=2000`, `seed=42`, 80:20 stratified split):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | **0.8600** | **0.8694** | **0.8773** | **0.8733** | **0.9373** |
| Random Forest | 0.8250 | 0.8378 | 0.8455 | 0.8416 | 0.9033 |
| XGBoost | 0.8275 | 0.8512 | 0.8318 | 0.8414 | 0.9095 |

## 🏆 Best Performing Model

**Logistic Regression** achieved the highest overall performance across every metric on
this dataset:

- **Accuracy:** 86.00%
- **F1-Score:** 87.33%
- **ROC-AUC:** 93.73%

This is expected given the dataset's construction: the synthetic "placed" label is a
**thresholded linear combination** of the input features, which is exactly the
relationship a linear model is best suited to recover — Random Forest and XGBoost are
included specifically to demonstrate model comparison discipline and would be expected
to pull ahead on real-world data with non-linear feature interactions (e.g. diminishing
returns on CGPA, threshold effects in DSA scores). The Keras ANN (Module 8) is trained
and evaluated separately as a fourth point of comparison in the notebook.

**Top features by XGBoost importance:** `projects_count` (0.152) → `internship_experience`
(0.150) → `dsa_score` (0.147) → `aptitude_score` (0.129) → `cgpa` (0.110) —
practical experience and technical scores outweigh even CGPA in this dataset, which
mirrors commonly cited industry hiring feedback.

---

## ▶️ How to Run

1. Open the notebook in **Google Colab**.
2. Run the install cell (Step 0) — it installs all dependencies and patches `Pillow`.
3. **Runtime → Restart session** (required once, immediately after the install cell —
   do *not* re-run the install cell afterward).
4. Run all remaining cells in order.
5. When prompted, paste a free **Gemini API key** from
   [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
6. Scroll to **Module 10** to launch the Gradio app, or call any module's function
   directly (`analyze_resume(...)`, `calculate_ats_score(...)`, `predict_placement(...)`,
   `rag_answer(...)`, `run_mock_interview(...)`) for standalone testing.

---

## 💡 Reason for Choosing This Project

Placement preparation is a problem nearly every engineering student faces personally,
which made it an immediately motivating domain to apply AI to. It was also chosen
because it deliberately spans the **full modern AI stack** in one project — classical
ML classification, deep learning, NLP-based document parsing, retrieval-augmented
generation, and LLM-based generative reasoning — rather than demonstrating just one
technique, making it a strong single artifact to showcase end-to-end AI engineering
competency.

## 📚 What I Learned

- How to design one cohesive system out of otherwise-separate ML, DL, NLP, and RAG
  components, with shared state (e.g. one scaler, one skill taxonomy) reused correctly
  across modules.
- How to combine **TF-IDF, rule-based skill matching, and semantic embeddings** into a
  single interpretable score (ATS) rather than relying on any one signal alone.
- Why **model comparison** (Logistic Regression vs. Random Forest vs. XGBoost vs. a
  Keras ANN) matters even when a simpler model wins — the goal is evidence, not a
  favorite algorithm.
- How to build a **RAG pipeline from scratch** (chunking → embedding → FAISS indexing →
  retrieval → grounded LLM generation) instead of treating it as a black box.
- How to **prompt an LLM for structured, parseable JSON output** reliably, with
  fallback handling for when generation doesn't strictly follow the schema.
- How to wrap retry/backoff/error-handling around a third-party LLM API so a transient
  failure doesn't crash the whole pipeline.
- How to package many independent modules into **one unified, multi-tab Gradio app**
  that a non-technical user could actually operate.

## ✅ Conclusion

This project successfully delivers a complete, multi-technique AI system for placement
preparation — covering the full pipeline from raw resume PDF to a personalized learning
roadmap and a scored mock interview. By combining classical ML, deep learning, NLP, and
a from-scratch RAG pipeline grounded with Gemini 3.5 Flash, the project goes beyond a
single-technique exercise and reflects an industry-style applied AI workflow suitable
for a portfolio, GitHub showcase, or technical interview discussion.

## ⚠️ Special Notes

- **Not a substitute for career counseling:** This project is for educational/portfolio
  purposes only. Placement probability, ATS scores, and mock interview feedback are
  model-generated estimates, not guarantees — always seek guidance from real placement
  cells, mentors, and recruiters.
- **Synthetic data:** The placement dataset is programmatically generated
  (`seed=42`, fully reproducible) rather than collected from real institutional
  records; results will shift if trained on real historical placement data.
- **API key security:** Never hardcode your `GEMINI_API_KEY` — the notebook reads it via
  `getpass`/environment variable only. Get a free key at
  [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
- **Restart requirement:** Skipping the **Runtime → Restart session** step after the
  install cell will reliably cause a `Pillow`/`torchvision` import error — this is a
  known Colab quirk, not a bug in the code.
- **LLM output variability:** Roadmap generation, RAG answers, and mock-interview
  scoring are generative — outputs will vary slightly between runs even with the same
  input, unlike the deterministic ML/DL modules.
- Remember to replace every `[bracketed placeholder]` in this README (Internship
  Details, Candidate Details, Colab link) with your actual information before
  submission.

---

## 📄 License

This project is released under the **MIT License** — feel free to use, modify, and
build upon it with attribution.

## 🙏 Acknowledgments

- Built using open-source tools: Python, Pandas, NumPy, Scikit-learn, XGBoost,
  TensorFlow/Keras, spaCy, Sentence-Transformers, FAISS, Matplotlib, Seaborn, Plotly,
  Gradio, and ReportLab.
- LLM capabilities powered by **Google Gemini 3.5 Flash** via the Gemini API.

---

<div align="center">

# 🚀 Kanishka Pal

### 🤖 AI/ML Enthusiast | 💻 Full-Stack Learner

<p>
  <a href="mailto:kksnehapal@gmail.com">📧 Email</a> •
  <a href="https://www.linkedin.com/in/kanishkapal2005/">💼 LinkedIn</a> •
</p>

*Made with care during my internship — building AI tools for better placement outcomes.*

</div>
