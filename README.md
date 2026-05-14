# Legal Case Outcome Prediction

This project is an NLP-based machine learning system for predicting the outcome of legal court cases based on judgement text.

The main idea of the project is to use legal text data and machine learning models to estimate whether a case outcome is likely to be positive or negative. The project also includes explainability components, such as LIME explanations and a simple legal norms layer.

## Project Overview

The system uses court case texts as input and produces:

- predicted case outcome;
- probability score;
- confidence level;
- explanation of important words or phrases;
- possible connection to legal norms.

The project was created as an experimental legal AI prototype. It is not intended to replace lawyers or judges. The prediction should be treated only as decision support.

## Main Features

- Text preprocessing for legal judgement documents
- TF-IDF + Logistic Regression baseline model
- Student model for judgement-based prediction
- Chunked judgement processing for long legal texts
- FastAPI backend
- Simple web interface
- LIME-based explanation module
- Legal norms matching layer
- Abstain / uncertainty mode for low-confidence predictions

## Project Structure

```text
legal_prediction/
├── api/
│   └── main.py                  # FastAPI backend
├── static/
│   └── index.html               # Simple frontend page
├── src/
│   ├── explain_lime.py          # LIME explanation logic
│   ├── legal_layer.py           # Legal norms matching
│   ├── final_explain.py         # Final explanation builder
│   └── evaluate_chunked_judgment.py
├── legal_dict/
│   └── legal_norms.csv          # Manually prepared legal norms
├── models/
│   ├── baseline_tfidf_logreg.joblib
│   ├── best_student_chunked_judgment.pt
│   ├── best_student_proof_alpha08.pt
│   ├── baseline_metrics.json
│   ├── student_metrics.json
│   └── chunked_threshold.json
├── raw/                         # Raw data, not uploaded to GitHub
├── splits/                      # Train/validation/test splits
├── README.md
└── requirements.txt