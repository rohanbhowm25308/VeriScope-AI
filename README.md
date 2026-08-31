# VeriScope AI

AI-Powered Claim Verification & Evidence Intelligence Platform

VeriScope AI is an advanced Machine Learning and Natural Language Processing (NLP) project designed to identify claims that require external verification. It analyzes claims based on context sufficiency, linguistic patterns, uncertainty, temporal sensitivity, and risk factors, then uses evidence retrieval and conflict analysis to provide an explainable verification recommendation.

The system is designed around the idea that an AI should not blindly classify every claim as true or false. When the available context is insufficient or the model is uncertain, VeriScope AI can abstain and recommend external verification or human review.

# 🌐 Live Demo

Live Application:
https://veriscope-ai-1.onrender.com

GitHub Repository:
https://github.com/rohanbhowm25308/VeriScope-AI

# ✨ Features
 Automatic Claim Extraction — Extracts potentially verifiable claims from user-provided text.
 Compound Claim Decomposition — Breaks complex sentences containing multiple claims into individual claims.
 Verification Requirement Detection — Predicts whether a claim requires external verification.
 Context Sufficiency Analysis — Determines whether the available context provides enough information to assess a claim.
 AI Abstention Mode — Allows the model to abstain when prediction confidence is insufficient.
 Claim Risk Scoring — Estimates the potential verification priority of a claim.
 Temporal Sensitivity Detection — Identifies claims that may change over time.
 Evidence Retrieval — Retrieves relevant evidence using information-retrieval techniques.
 Evidence Conflict Detection — Identifies potentially supporting and contradicting evidence.
 Explainable AI — Provides reasons behind verification recommendations.
 Multi-Model ML Comparison — Supports experimentation with multiple machine-learning models.
 Human-in-the-Loop Review — Allows uncertain or high-priority claims to be reviewed by humans.
 Analytics & Evaluation — Provides model-performance and claim-analysis insights.
 Verification Reports — Generates structured verification summaries.
 Error Analysis — Helps identify cases where the model makes incorrect or uncertain predictions.
 
 # How It Works
 
User Input
    ↓
Claim Extraction
    ↓
Compound Claim Decomposition
    ↓
NLP Feature Analysis
    ↓
Context Sufficiency Analysis
    ↓
ML Verification Prediction
    ↓
Confidence & Risk Analysis
    ↓
AI Abstention Check
    ↓
Evidence Retrieval
    ↓
Evidence Conflict Analysis
    ↓
Explainable Result
    ↓
Human Review / Verification

#  Machine Learning

VeriScope AI uses NLP and machine-learning techniques to analyze textual claims. The system can experiment with models such as:

Logistic Regression
Random Forest
Linear SVM
Multinomial Naive Bayes

Text is transformed into numerical representations using TF-IDF, along with engineered linguistic features for claim analysis.

The project also includes model evaluation and error-analysis workflows to understand model behavior rather than relying only on a single accuracy score.

# Evidence Intelligence

The evidence-analysis component uses information-retrieval techniques to find relevant context for a claim.

The system can analyze:

Evidence relevance
Semantic/context similarity
Evidence strength
Potentially conflicting information
Context coverage
Verification priority

This allows VeriScope AI to move beyond simple text classification toward an evidence-aware verification workflow.

 AI Abstention

One of the key features of VeriScope AI is its ability to abstain from making an unreliable prediction.

Instead of forcing the model to select a class when confidence is low:

Low Confidence
      ↓
AI ABSTAINS
      ↓
External Verification Recommended
      ↓
Human / Evidence Review

This makes the system more suitable for situations where an incorrect automated decision could be misleading.

 Verification Categories

The system can categorize claims according to their verification requirements, such as:

 Context Sufficient
 Needs Verification
 High-Priority Verification
 AI Abstained

The objective is not to automatically declare a claim true or false, but to determine whether additional evidence is necessary before the claim can be responsibly accepted.

#  Tech Stack

Languages:
Python, JavaScript, HTML, CSS

Machine Learning:
Scikit-learn, NumPy, SciPy, Pandas

NLP:
NLTK, TF-IDF, linguistic feature engineering

Information Retrieval:
Rank-BM25, cosine similarity, evidence ranking

Backend:
Flask, Flask-CORS, Gunicorn

Frontend:
HTML, CSS, JavaScript

Deployment:
GitHub, Render

 Installation

Clone the repository:

git clone https://github.com/rohanbhowm25308/VeriScope-AI.git

Move into the backend directory:

cd VeriScope-AI/backend

Create a virtual environment:

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

Install the dependencies:

pip install -r requirements.txt

Run the application:

python app.py

Then open:

http://localhost:5000
 Deployment

VeriScope AI is deployed on Render.

Render Configuration
Root Directory:
backend

Build Command:
pip install -r requirements.txt

Start Command:
gunicorn app:app
Live Website

https://veriscope-ai-3eyf.onrender.com/

# Research Focus

VeriScope AI is based on the research problem of determining whether an NLP system can recognize when a claim cannot responsibly be accepted without external evidence.

The project focuses on:

Claim detection
Verification-need prediction
Context sufficiency
Uncertainty estimation
Evidence retrieval
Evidence conflict
Explainable AI
Human-in-the-loop verification
Model evaluation and error analysis

 Limitations

VeriScope AI is an experimental machine-learning prototype. Its predictions depend on the quality and coverage of the training data, available context, retrieval results, and model behavior. Therefore, the system should be considered a verification-prioritization and decision-support tool, not an autonomous source of truth.

# 👨‍💻 Author

Rohan Bhowmik
B.Tech — Computer Science & Engineering

Interests: Machine Learning · Artificial Intelligence · Data Science · NLP · Web Development

⭐ Support

If you find VeriScope AI useful or interesting, consider giving the repository a ⭐ on GitHub.

VeriScope AI — Analyze the claim. Understand the context. Find the evidence.
