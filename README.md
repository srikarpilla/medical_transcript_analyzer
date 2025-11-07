Medical Transcript Analyzer
This project runs a cool NLP pipeline to analyze medical conversations between doctors and patients. It pulls out symptoms, treatments, and gives a summary, sentiment check, and SOAP notes from the transcript. It's especially handy for quick prototyping of medical dialogue AI.

Features
Extracts and summarizes medical entities (like symptoms, diagnosis, etc.)

Checks patient sentiment and main intent

Builds a quick medical summary

Generates classic SOAP (Subjective, Objective, Assessment, Plan) notes

Tech Used
Python 3.10

Transformers (for sentiment analysis with distilBERT)

spaCy (NER tasks)

scikit-learn (TF-IDF keyword extraction)

NLTK (sentence tokenizing)














Setup Instructions
Clone this repo or copy the code

Install dependencies:


pip install spacy torch transformers sklearn nltk


python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt')"


Run:


pip install -r requirements.txt
python -m spacy download en_core_web_sm

streamlit run app2.py


Note:
The default sample demonstrates how everything works; you can plug in your own medical dialogue.


