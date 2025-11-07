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

(Optional) It’s better to make a virtual environment:


python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
Install dependencies:


pip install spacy torch transformers sklearn nltk
Download the spaCy model and nltk punkt (automatic in code, but just in case):


python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt')"


Run:


pip install -r requirements.txt
python -m spacy download en_core_web_sm

streamlit run app2.py


Notes
The default sample demonstrates how everything works; you can plug in your own medical dialogue.

Modify the regex patterns in the code for different symptoms or diagnosis rules.

Output includes JSON for summaries and standard printouts.

This is just a prototype. For real clinical use, further validation is definitely needed.

