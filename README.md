# AI-Generated Text Detection

### English AI-Generated Text Detection with Machine Learning

This repository contains the implementation, experiments, and frontend demo for an English-focused AI-generated text detection project.

The current application supports English text only. It uses a full-text essay detector for the overall AI percentage and an English sentence detector to rank local sections for highlighting.

---

## Project Paper

The full research paper is included in this repository as a PDF:

Project Research.pdf

Authors:

Adham Mohamed  
Ammar Alaa  
Youssef Mohamed Abdelshahid  
Youssef Mohamed Amer  

Faculty of Computer Science, Misr International University, Cairo, Egypt

---

## Datasets Used

1. English Essays Dataset
   - Long, structured essay-style texts
   - Labels: 0 = Human, 1 = AI
   - Used to evaluate performance on formal writing

2. English Sentences Dataset
   - Short, casual texts from:
     - Twitter
     - Reddit
     - AI Text Detection Pile (Hugging Face)
   - Balanced human vs AI samples
   - Designed for real-world conversational language

---

## Preprocessing Pipeline

Text Cleaning
-> Tokenization
-> Stopword Removal
-> Lemmatization
-> Train-Test Split (70 / 30)
-> Vectorization (TF-IDF or Tokenizer)

English preprocessing uses NLTK stopwords and lemmatization.

---

## Models Implemented

Machine Learning Models:
- Logistic Regression
- Random Forest
- Naive Bayes
- Passive Aggressive
- K-Nearest Neighbors
- XGBoost
- Feature extraction using TF-IDF

Deep Learning Models:
- Deep Neural Network
- Bidirectional LSTM
- Gated Recurrent Unit
- Convolutional Neural Network

Transformer Models:
- RoBERTa
- ELECTRA

---

## Evaluation Metrics

Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC

Deep learning and transformer-based models outperform traditional machine learning approaches on complex English text detection tasks.

---

## Notebooks

English Essays:
- Training and evaluation on long texts

English Sentences:
- Short informal text detection

Each notebook includes:
- Data loading
- Preprocessing
- Model training
- Evaluation
- Saving trained models

---

## Frontend Application

A Flask web application is provided to test trained English models.

Front-end/app.py

Features:
- Accepts English text input from the user
- Loads trained English models
- Predicts whether text is human-written or AI-generated
- Uses the English essay model for the main AI percentage and the English sentence model for local highlighting

---

## How to Run the Frontend Application

To launch the web-based demo locally, follow the steps below.

1. Navigate to the frontend directory:

   cd Front-end

2. Install dependencies from the project root if needed:

   pip install -r requirements.txt

3. Start the Flask application:

   python app.py

4. Open a web browser and visit:

   http://127.0.0.1:5000

The trained model files should be placed in the ignored `models/` directory:
- `models/sentences_passive_aggressive_pipeline.pkl`
- `models/essay_passive_aggressive_pipeline.pkl`

Processed dataset pickles such as `sentence.pkl` belong in the ignored `data/` directory, not `models/`.

The essay detector analyzes the full text as one document and provides the official AI percentage. The sentence detector only ranks local sections for highlighting, and the amount of highlighted text is limited by the essay model's overall score.

---

## Key Contributions

- English AI-text detection for essays and short texts
- Comparison of traditional machine learning, deep learning, and transformer models
- Deployable Flask frontend for live testing

---

## License

This project is intended for academic and research purposes.
Please cite the accompanying paper if you use or build upon this work.
