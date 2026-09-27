#  VeriScope AI

## AI-Powered Claim Verification & Evidence Auditor

> **Learn Depth — Problem ML-T2-061:** Detecting Claims That Require External Verification

> **Research Question:** Can an NLP system recognize when a claim cannot responsibly be accepted without external evidence?

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-VeriScope_AI-black?style=for-the-badge)](https://veriscope-ai-1.onrender.com/)
[![GitHub](https://img.shields.io/badge/💻_GitHub-Source_Code-black?style=for-the-badge&logo=github)](https://github.com/rohanbhowm25308/VeriScope-AI)

---

##  What is VeriScope AI?

**VeriScope AI** is an NLP and Machine Learning research prototype designed to identify claims that require external verification before they can be responsibly accepted.

Unlike conventional fake-news or truth-detection systems, VeriScope AI **does not attempt to declare a claim simply "True" or "False."**

Instead, it answers a more practical question:

> **"Does this claim have enough available context and evidence to be responsibly accepted?"**

The system analyzes claims using linguistic signals, TF-IDF representations, machine-learning models, context sufficiency, uncertainty estimation, risk scoring, evidence retrieval, conflict detection, and human-review routing.

---

##  Core Concept

```text
                    USER INPUT
                        │
                        ▼
              ┌───────────────────┐
              │  Claim Extraction │
              └─────────┬─────────┘
                        ▼
              ┌───────────────────┐
              │ Claim Classification│
              └─────────┬─────────┘
                        ▼
              ┌───────────────────┐
              │ Context Analysis  │
              └─────────┬─────────┘
                        ▼
              ┌───────────────────┐
              │ Uncertainty + Risk│
              └─────────┬─────────┘
                        ▼
              ┌───────────────────┐
              │ Evidence Retrieval│
              └─────────┬─────────┘
                        ▼
              ┌───────────────────┐
              │ Conflict Detection│
              └─────────┬─────────┘
                        ▼
              ┌───────────────────┐
              │ Human Review Queue│
              └───────────────────┘
```

---

# ✨ Key Features

###  Intelligent Claim Extraction
- Sentence segmentation
- Claimability filtering
- Opinion/question/belief detection
- Compound-claim decomposition
- Claim complexity analysis

###  Machine Learning Verification Routing
- TF-IDF feature representation
- Engineered linguistic features
- Check-worthiness prediction
- Multi-model classification
- Model consensus and disagreement detection
- Confidence estimation
- AI abstention

###  Evidence Intelligence
- BM25 evidence retrieval
- TF-IDF similarity
- Evidence ranking
- Evidence windowing
- Conflict detection
- Evidence freshness analysis
- Evidence intelligence scoring
- Optional web-search evidence path

###  Risk & Uncertainty Analysis
- Context sufficiency score
- Claim risk score
- Temporal sensitivity
- Claim type detection
- Claim fingerprint
- Adjustable verification threshold

###  Human-in-the-Loop
- Priority-based review queue
- Reviewer feedback
- Confidence and notes
- Claim lifecycle tracking
- Investigation roadmap
- Counterfactual testing

###  Research & Analytics
- Verification dashboard
- Model Comparison Laboratory
- Error analysis
- Feature importance
- Retrieval benchmark
- Research analytics
- Reviewed-case export
- Verification report generation

---

#  Verification Outcomes

| Verdict | Meaning |
|---|---|
|  **Context Sufficient** | Available context provides enough signal for responsible assessment |
|  **Needs Verification** | The claim is checkable but supporting evidence is currently unavailable |
|  **High Priority** | The claim requires verification and has elevated risk/stakes |
|  **AI Abstains** | Confidence is too low for an automated decision; routed to human review |
|  **Not a Claim** | Filtered as an opinion, question, or personal belief before verification |

---

#  End-to-End Pipeline

```text
Text Input
   │
   ▼
Sentence Segmentation
   │
   ▼
Claimability Filtering
   │
   ▼
Compound Claim Decomposition
   │
   ▼
Complexity Analysis
   │
   ▼
Linguistic Feature Engineering
   │
   ├── TF-IDF
   ├── Linguistic Signals
   └── Check-Worthiness Score
   │
   ▼
Hybrid ML + Rule-Based Decision
   │
   ▼
AI Abstention / Confidence Analysis
   │
   ▼
Context Sufficiency
   │
   ▼
Risk Scoring + Claim Fingerprint
   │
   ▼
Multi-Model Consensus
   │
   ▼
Evidence Retrieval
   │
   ├── BM25
   ├── TF-IDF Similarity
   └── Optional Web Search
   │
   ▼
Conflict Detection
   │
   ▼
Human Review / Investigation Roadmap
```

---

#  Machine Learning Architecture

VeriScope AI uses a combination of classical NLP, machine learning, and rule-based reasoning.

### Feature Representation

```text
                    Claim Text
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
          TF-IDF          Engineered Features
              │                   │
              └─────────┬─────────┘
                        ▼
                 ML Classifiers
                        │
                        ▼
              Consensus + Rules
                        │
                        ▼
                Final Routing
```

### Models

- Logistic Regression
- Random Forest
- Linear SVM
- Multinomial Naive Bayes
- LSA/SVD baseline
- Optional transformer-based experiment

The system also includes a dedicated **binary check-worthiness model** trained using real CheckThat! data.

---

#  Evaluation

The current research implementation reports:

| Evaluation | Result |
|---|---:|
| Binary Check-Worthiness ROC-AUC | ~0.74 |
| Binary Check-Worthiness PR-AUC | ~0.09 |
| 3-Class Linear SVM CV F1 Macro | ~0.76 ± 0.05 |
| Held-Out Test Accuracy | ~0.74 |
| Retrieval Precision@3 | 0.95 |
| Retrieval Recall@3 | 0.95 |

The retrieval benchmark was evaluated using a hand-labeled set of 20 cases.

---

#  Dataset

## Real Dataset

**CLEF CheckThat! 2019**

- Approximately 17,600 sentences
- US presidential debates and speeches
- 2016–2019 data
- Human-annotated
- Check-worthy vs. non-check-worthy labels
- Used to train the dedicated binary check-worthiness model

## Synthetic Seed Dataset

- 175 examples
- Roughly balanced across the three fine-grained verification categories
- Programmatically authored
- Covers difficult cases including:
  - Compound claims
  - Temporal claims
  - Conditional claims
  - Implicit claims

> The synthetic dataset is documented as a weak-supervision placeholder for a future fully human-annotated 3-way corpus.

---

# 🔬 Evidence Intelligence

VeriScope AI does not stop after classifying a claim.

It attempts to determine **what evidence should be examined next**.

### Evidence Pipeline

```text
Available Context
      │
      ▼
Evidence Candidate Generation
      │
      ▼
BM25 / Similarity Ranking
      │
      ▼
Relevant Evidence Selection
      │
      ▼
Atomic Sentence Comparison
      │
      ▼
Conflict Detection
      │
      ▼
Evidence Strength
      │
      ▼
Human Investigation
```

The system can identify conflicting numerical values, contradictory statements, and other evidence inconsistencies.

---

#  AI Abstention

One of the central research ideas behind VeriScope AI is:

> **An AI system should be able to say "I don't have enough confidence to make this call."**

Instead of forcing every claim into a definitive category, VeriScope AI can abstain when multiple conditions indicate insufficient confidence.

Abstention can consider:

- Model confidence
- Context sufficiency
- Evidence availability
- Model disagreement
- Claim complexity
- Risk level

The uncertain case is then routed toward human investigation.

---

#  Model Comparison Laboratory

The application includes a dedicated model comparison environment where different classical ML approaches can be evaluated using the same feature representation.

It provides:

- Test accuracy
- Precision
- Recall
- F1 score
- Cross-validation results
- Error analysis
- Feature importance
- Model disagreement
- Check-worthiness model comparison

This makes the project not only an application, but also an **experimental ML research platform**.

---

#  Research Analytics Dashboard

The Research Analytics Dashboard tracks:

- Stored claim history
- Reviewer feedback
- Verification outcomes
- Feature importance
- Retrieval benchmark results
- Model behavior
- Human-review activity
- Research statistics

Reviewed cases can also be exported for future retraining and experimentation.

---

#  Human-in-the-Loop Architecture

```text
                 AI ANALYSIS
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
      High Risk              Low Confidence
          │                       │
          └───────────┬───────────┘
                      ▼
              Human Review Queue
                      │
             ┌────────┴────────┐
             ▼                 ▼
        Reviewer Notes     Feedback
             │                 │
             └────────┬────────┘
                      ▼
              Research Analytics
                      │
                      ▼
               Future Retraining
```

---

#  Technology Stack

## Backend
- Python
- Flask
- Flask-CORS
- Gunicorn
- SQLite

## Machine Learning
- Scikit-learn
- TF-IDF
- Linear SVM
- Logistic Regression
- Random Forest
- Multinomial Naive Bayes
- LSA/SVD

## NLP & Evidence
- NLP preprocessing
- Regex/rule-based claim extraction
- BM25
- TF-IDF similarity
- Evidence ranking
- Conflict detection

## Frontend
- HTML5
- CSS3
- JavaScript

## Deployment
- GitHub
- Render
- Optional external/web-search integration

```

---

#  Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/rohanbhowm25308/VeriScope-AI.git
cd VeriScope-AI
```

### 2. Navigate to Backend

```bash
cd backend
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure Environment Variables

```bash
cp .env.example .env
```

Add the optional API configuration if required.

### 7. Start the Application

```bash
python app.py
```

The application will be available at:

```text
http://localhost:5000
```

---

#  Retraining Models

The repository contains scripts for rebuilding the datasets and models.

```bash
python data/prepare_checkthat_data.py
python data/build_seed_dataset_v2.py
python train_checkworthy_model.py
python train_models.py
python eval_retrieval_methods.py
```

An optional transformer experiment is also included:

```bash
python train_models_transformer_experiment.py
```

---

#  Deployment

The project is deployed using GitHub and Render.

### Live Application

https://veriscope-ai-1.onrender.com/

### Source Code

https://github.com/rohanbhowm25308/VeriScope-AI

For a Render deployment, the Flask backend can be configured with:

```text
Root Directory: backend
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app
```

---

#  Engineering Challenges

During development, several real-world issues were identified and fixed.

### 1. BM25 Negative-IDF Problem

Small candidate pools caused BM25 scores to become negative and hide relevant evidence.

**Solution:** Per-term IDF flooring was introduced.

### 2. Stopword False Positives

Unrelated sentences could receive strong similarity scores simply because they shared common words.

**Solution:** Stopwords were removed before BM25 indexing and minimum genuine term overlap was required.

### 3. Evidence Window Conflict Masking

Evidence windows containing contradictory numerical values could hide the underlying conflict.

**Solution:** Evidence windows were decomposed back into atomic sentences before conflict comparison.

### 4. External Search Integration

The external web-search integration required migration after the underlying Groq model used by the earlier implementation was retired.

The project now uses a graceful fallback architecture rather than exposing a raw API failure.

---

#  Known Limitations

VeriScope AI is a research prototype, not a production-grade fact-checking authority.

Current limitations include:

- The fine-grained 3-way dataset contains approximately 175 weakly labeled/programmatically generated examples.
- The check-worthiness model has modest precision/recall because the underlying task is highly imbalanced and difficult.
- Local evidence retrieval depends on the context supplied by the user.
- Claim extraction uses lightweight regex/rule-based processing rather than a dependency parser.
- Full claim-relationship graph visualization is not yet implemented.
- Claim-change tracking/versioning is future work.
- PDF/DOCX upload is not currently implemented.
- SQLite persistence on the free Render environment is not suitable for long-term production storage.

---

#  What Makes VeriScope AI Different?

### 1. It Does Not Pretend to Know Everything

Instead of forcing a True/False answer, the system can identify uncertainty and abstain.

### 2. It Focuses on Verification Routing

The goal is to determine:

```text
What should be accepted?
What needs evidence?
What needs urgent investigation?
What should be reviewed by a human?
```

### 3. It Combines ML + Explainable Rules

The system combines:

```text
Machine Learning
        +
Linguistic Features
        +
Context Analysis
        +
Risk Scoring
        +
Evidence Retrieval
        +
Conflict Detection
        +
Human Review
```

### 4. It Includes an Actual Research Workflow

The project includes:

- Model comparison
- Error analysis
- Retrieval benchmarking
- Feature importance
- Counterfactual testing
- Reviewer feedback
- Research analytics
- Verification reports

---

# 🎓 Internship / Problem Information

**Program:** Advanced Machine Learning Internship  
**Organization:** Learn Depth Academy LLP  
**Problem ID:** ML-T2-061  
**Problem:** Detecting Claims That Require External Verification  
**Domain:** NLP · Misinformation Research  
**Level:** Advanced  
**Project:** VeriScope AI

### Research Question

> **Can an NLP system recognize when a claim cannot responsibly be accepted without external evidence?**

---

# 👨‍💻 Developer

**Rohan Bhowmik**

B.Tech Computer Science & Engineering

### Areas of Interest

- Artificial Intelligence
- Machine Learning
- Data Science
- Natural Language Processing
- Generative AI
- Web Development
- Python
- Explainable AI

---

#  Feedback & Contribution

If you explore VeriScope AI, feedback is welcome.

Ideas, research suggestions, model improvements, retrieval strategies, and usability feedback can help improve the system further.

If you find the project interesting, consider giving the repository a ⭐ on GitHub.

---

#  Disclaimer

VeriScope AI is an experimental research prototype designed to assist with claim auditing and verification routing.

It does **not** establish objective truth or falsehood and should not be treated as an authoritative fact-checking system.

Its recommendations should be interpreted as signals for further investigation and, where appropriate, human review.

---

<div align="center">

###  VeriScope AI

**"Don't just ask whether a claim is true. Ask whether you have enough evidence to accept it."**

Built with Python • NLP • Machine Learning • Evidence Intelligence

**Developed by Rohan Bhowmik**

</div>
