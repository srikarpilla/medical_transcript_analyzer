"""
Medical Transcript NLP Analyzer
AI/ML Internship Assignment
Author: [Your Name]

This app does medical transcription analysis with NER, sentiment analysis, and SOAP note generation.
I tried different approaches and documented what worked vs what didn't.
"""

import streamlit as st
import json
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Core NLP imports
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)


class MedicalNLPEngine:
    """
    Main NLP engine for medical transcript processing.
    Handles NER, summarization, keyword extraction, and entity recognition.
    """
    
    def __init__(self):
        """Initialize NLP models. This takes a while on first run."""
        print("🔄 Loading NLP models (this might take a minute)...")
        
        # Load spaCy - tried en_core_web_trf but too slow for demo
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("📥 Downloading spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Add custom medical entity ruler
        # This helps with domain-specific terms that spaCy might miss
        if "entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            patterns = self._get_medical_patterns()
            ruler.add_patterns(patterns)
        
        # Sentiment analysis - using distilbert (smaller, faster than BERT)
        # TODO: Fine-tune on medical data if time permits
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1  # CPU (use 0 for GPU if available)
        )
        
        # Medical keyword dictionary for better extraction
        self.medical_vocab = self._build_medical_vocab()
        
        print("✅ Models loaded successfully!")
    
    def _get_medical_patterns(self) -> List[Dict]:
        """
        Custom patterns for medical entity recognition.
        Helps catch domain-specific terms.
        """
        patterns = [
            # Diagnoses
            {"label": "DIAGNOSIS", "pattern": [{"LOWER": "whiplash"}]},
            {"label": "DIAGNOSIS", "pattern": [{"LOWER": "whiplash"}, {"LOWER": "injury"}]},
            {"label": "DIAGNOSIS", "pattern": [{"LOWER": "soft"}, {"LOWER": "tissue"}, {"LOWER": "injury"}]},
            {"label": "DIAGNOSIS", "pattern": [{"LOWER": "strain"}]},
            {"label": "DIAGNOSIS", "pattern": [{"LOWER": "sprain"}]},
            
            # Treatments
            {"label": "TREATMENT", "pattern": [{"LOWER": "physiotherapy"}]},
            {"label": "TREATMENT", "pattern": [{"LOWER": "physical"}, {"LOWER": "therapy"}]},
            {"label": "TREATMENT", "pattern": [{"LOWER": "painkillers"}]},
            {"label": "TREATMENT", "pattern": [{"LOWER": "pain"}, {"LOWER": "medication"}]},
            {"label": "TREATMENT", "pattern": [{"LOWER": "analgesics"}]},
            
            # Symptoms
            {"label": "SYMPTOM", "pattern": [{"LOWER": "neck"}, {"LOWER": "pain"}]},
            {"label": "SYMPTOM", "pattern": [{"LOWER": "back"}, {"LOWER": "pain"}]},
            {"label": "SYMPTOM", "pattern": [{"LOWER": "backache"}]},
            {"label": "SYMPTOM", "pattern": [{"LOWER": "stiffness"}]},
            {"label": "SYMPTOM", "pattern": [{"LOWER": "discomfort"}]},
            {"label": "SYMPTOM", "pattern": [{"LOWER": "tenderness"}]},
        ]
        return patterns
    
    def _build_medical_vocab(self) -> set:
        """Build medical vocabulary for keyword extraction."""
        return {
            'pain', 'ache', 'discomfort', 'hurt', 'sore', 'stiff', 'tender',
            'neck', 'back', 'head', 'spine', 'muscle', 'joint',
            'whiplash', 'injury', 'strain', 'sprain', 'trauma',
            'physiotherapy', 'therapy', 'treatment', 'medication', 'painkiller',
            'recovery', 'healing', 'prognosis', 'diagnosis',
            'accident', 'collision', 'impact',
            'examination', 'assessment', 'evaluation'
        }
    
    def extract_entities_spacy(self, text: str) -> Dict[str, List[str]]:
        """
        Use spaCy NER to extract medical entities.
        This is the proper NLP way vs pure regex.
        """
        doc = self.nlp(text)
        
        entities = {
            "symptoms": [],
            "diagnosis": [],
            "treatment": [],
            "body_parts": [],
            "dates": [],
            "facilities": []
        }
        
        # Extract custom entities from entity ruler
        for ent in doc.ents:
            if ent.label_ == "SYMPTOM":
                entities["symptoms"].append(ent.text)
            elif ent.label_ == "DIAGNOSIS":
                entities["diagnosis"].append(ent.text)
            elif ent.label_ == "TREATMENT":
                entities["treatment"].append(ent.text)
            elif ent.label_ == "DATE":
                entities["dates"].append(ent.text)
            elif ent.label_ == "ORG" or ent.label_ == "FAC":
                # Organizations/Facilities (hospitals, clinics)
                entities["facilities"].append(ent.text)
        
        # Also use regex as fallback for things spaCy might miss
        # (Not ideal but works for demo)
        entities = self._augment_with_regex(text, entities)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def _augment_with_regex(self, text: str, entities: Dict) -> Dict:
        """
        Fallback regex extraction for things NER might miss.
        This handles the 'ambiguous data' question from requirements.
        """
        text_lower = text.lower()
        
        # Symptoms - common patterns
        symptom_patterns = [
            r'(neck|back|head)\s+(pain|ache|hurt)',
            r'trouble\s+sleeping',
            r'(stiff|sore|tender)',
            r'(occasional|constant|severe)\s+(pain|discomfort)'
        ]
        for pattern in symptom_patterns:
            matches = re.finditer(pattern, text_lower)
            for m in matches:
                entities["symptoms"].append(m.group(0))
        
        # Treatments with numbers (e.g., "10 sessions of physiotherapy")
        treatment_pattern = r'(\d+)\s+sessions?\s+of\s+(\w+)'
        for m in re.finditer(treatment_pattern, text_lower):
            entities["treatment"].append(f"{m.group(1)} {m.group(2)} sessions")
        
        return entities
    
    def extract_keywords_tfidf(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Extract keywords using TF-IDF.
        Returns keywords with their scores.
        """
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) < 2:
            # Not enough text for TF-IDF
            return self._fallback_keyword_extraction(text, top_n)
        
        try:
            # TF-IDF with medical vocabulary boost
            vectorizer = TfidfVectorizer(
                max_features=top_n * 2,
                stop_words='english',
                ngram_range=(1, 3),  # unigrams to trigrams
                vocabulary=self.medical_vocab if len(self.medical_vocab) > 0 else None
            )
            
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF score for each term
            scores = tfidf_matrix.toarray().mean(axis=0)
            
            # Sort by score
            keyword_scores = [(feature_names[i], scores[i]) for i in range(len(feature_names))]
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return keyword_scores[:top_n]
        
        except Exception as e:
            print(f"⚠️ TF-IDF failed: {e}, using fallback")
            return self._fallback_keyword_extraction(text, top_n)
    
    def _fallback_keyword_extraction(self, text: str, top_n: int) -> List[Tuple[str, float]]:
        """Fallback to simple frequency-based extraction if TF-IDF fails."""
        words = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        
        # Filter medical terms
        medical_words = [w for w in words if w in self.medical_vocab and w not in stop_words]
        
        # Count frequencies
        freq = Counter(medical_words)
        
        # Return top N with pseudo-scores
        return [(word, count / len(medical_words)) for word, count in freq.most_common(top_n)]
    
    def parse_conversation(self, transcript: str) -> Tuple[List[str], List[str]]:
        """
        Parse transcript into doctor and patient statements.
        Handles different formatting styles.
        """
        lines = transcript.strip().split('\n')
        doctor_lines = []
        patient_lines = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('['):  # Skip empty or stage directions
                continue
            
            # Handle different formats: "Doctor:", "Physician:", "**Physician:**"
            if re.match(r'(\*\*)?([Pp]hysician|[Dd]octor)(\*\*)?:', line):
                text = re.sub(r'\*\*?([Pp]hysician|[Dd]octor)\*?:', '', line)
                text = text.strip('* \t')
                if text:
                    doctor_lines.append(text)
            
            elif re.match(r'(\*\*)?[Pp]atient(\*\*)?:', line):
                text = re.sub(r'\*\*?[Pp]atient\*?:', '', line)
                text = text.strip('* \t')
                if text:
                    patient_lines.append(text)
        
        return doctor_lines, patient_lines


class SentimentAnalyzer:
    """
    Handles sentiment and intent analysis for patient statements.
    Addresses the 'fine-tuning BERT for medical sentiment' question.
    """
    
    def __init__(self, sentiment_pipeline):
        self.sentiment_pipeline = sentiment_pipeline
        
        # Intent classification rules
        # In production, this would be a trained classifier
        self.intent_patterns = {
            "seeking_reassurance": [
                r'\b(will i|should i|worried|afraid|anxious|concern)\b',
                r'\?.*\b(better|recover|heal)\b'
            ],
            "reporting_symptoms": [
                r'\b(pain|hurt|ache|feel|feeling)\b',
                r'\b(trouble|difficulty|problem)\b'
            ],
            "expressing_concern": [
                r'\b(worried|concerned|afraid|anxious|nervous)\b',
                r'\b(future|long.?term|permanent)\b'
            ]
        }
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of patient statement.
        
        Returns:
            dict with sentiment, intent, confidence
        """
        # Truncate to model's max length
        text_truncated = text[:512]
        
        try:
            result = self.sentiment_pipeline(text_truncated)[0]
            raw_sentiment = result['label']  # POSITIVE or NEGATIVE
            confidence = result['score']
            
            # Map to medical context
            # This is a heuristic approach - ideally would fine-tune on medical data
            if raw_sentiment == 'NEGATIVE' and confidence > 0.65:
                sentiment = "Anxious"
            elif raw_sentiment == 'POSITIVE' and confidence > 0.65:
                sentiment = "Reassured"
            else:
                sentiment = "Neutral"
            
            # Detect intent
            intent = self._detect_intent(text)
            
            return {
                "sentiment": sentiment,
                "intent": intent,
                "confidence": round(confidence, 3),
                "raw_sentiment": raw_sentiment
            }
        
        except Exception as e:
            print(f"⚠️ Sentiment analysis error: {e}")
            return {
                "sentiment": "Neutral",
                "intent": "General communication",
                "confidence": 0.5,
                "raw_sentiment": "UNKNOWN"
            }
    
    def _detect_intent(self, text: str) -> str:
        """Rule-based intent detection."""
        text_lower = text.lower()
        
        # Check each intent pattern
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return intent.replace('_', ' ').title()
        
        return "General Communication"


class MedicalSummarizer:
    """Generates structured medical summaries from transcripts."""
    
    def __init__(self, nlp_engine: MedicalNLPEngine):
        self.nlp = nlp_engine
    
    def generate_summary(self, transcript: str) -> Dict:
        """
        Generate structured medical summary.
        This answers the 'handling ambiguous/missing data' question.
        """
        doctor_lines, patient_lines = self.nlp.parse_conversation(transcript)
        full_text = ' '.join(doctor_lines + patient_lines)
        patient_text = ' '.join(patient_lines)
        
        # Extract entities using NER
        entities = self.nlp.extract_entities_spacy(full_text)
        
        # Extract keywords
        keywords = self.nlp.extract_keywords_tfidf(full_text, top_n=8)
        keyword_list = [kw[0] for kw in keywords]
        
        # Build structured summary
        summary = {
            "patient_name": self._extract_patient_name(full_text),
            "symptoms": entities["symptoms"] if entities["symptoms"] else self._extract_symptoms_fallback(full_text),
            "diagnosis": entities["diagnosis"][0] if entities["diagnosis"] else self._extract_diagnosis_fallback(full_text),
            "treatment": entities["treatment"] if entities["treatment"] else self._extract_treatment_fallback(full_text),
            "current_status": self._extract_current_status(patient_text),
            "prognosis": self._extract_prognosis(full_text),
            "keywords": keyword_list,
            "dates": entities["dates"],
            "facilities": entities["facilities"],
            "data_quality": self._assess_data_quality(entities)
        }
        
        return summary
    
    def _extract_patient_name(self, text: str) -> str:
        """Extract patient name from text."""
        # Look for "Mr./Ms./Mrs. [Name]"
        match = re.search(r'\b(Mr|Ms|Mrs)\.?\s+([A-Z][a-z]+)\b', text)
        if match:
            return f"{match.group(1)}. {match.group(2)}"
        return "Patient"
    
    def _extract_symptoms_fallback(self, text: str) -> List[str]:
        """Fallback symptom extraction if NER fails."""
        symptoms = []
        text_lower = text.lower()
        
        if 'neck pain' in text_lower or 'neck hurt' in text_lower:
            symptoms.append("Neck pain")
        if 'back pain' in text_lower or 'backache' in text_lower:
            symptoms.append("Back pain")
        if 'head' in text_lower and any(w in text_lower for w in ['hit', 'struck', 'impact']):
            symptoms.append("Head impact")
        if 'trouble sleeping' in text_lower or 'difficulty sleeping' in text_lower:
            symptoms.append("Sleep disturbance")
        if 'stiff' in text_lower:
            symptoms.append("Stiffness")
        
        return symptoms if symptoms else ["Pain and discomfort (unspecified)"]
    
    def _extract_diagnosis_fallback(self, text: str) -> str:
        """Fallback diagnosis extraction."""
        text_lower = text.lower()
        if 'whiplash' in text_lower:
            return "Whiplash injury"
        elif 'strain' in text_lower:
            return "Soft tissue strain"
        elif 'sprain' in text_lower:
            return "Sprain"
        else:
            return "Soft tissue injury (unspecified)"
    
    def _extract_treatment_fallback(self, text: str) -> List[str]:
        """Fallback treatment extraction."""
        treatments = []
        text_lower = text.lower()
        
        # Look for physiotherapy with session count
        match = re.search(r'(\d+)\s+sessions?\s+of\s+physiotherapy', text_lower)
        if match:
            treatments.append(f"{match.group(1)} physiotherapy sessions")
        elif 'physiotherapy' in text_lower:
            treatments.append("Physiotherapy")
        
        if any(word in text_lower for word in ['painkiller', 'pain medication', 'analgesic']):
            treatments.append("Pain medication")
        
        return treatments if treatments else ["Conservative management"]
    
    def _extract_current_status(self, patient_text: str) -> str:
        """Extract current symptom status."""
        text_lower = patient_text.lower()
        
        if 'occasional' in text_lower and 'pain' in text_lower:
            return "Occasional pain"
        elif 'better' in text_lower or 'improving' in text_lower:
            return "Improving"
        elif 'worse' in text_lower or 'worsening' in text_lower:
            return "Worsening"
        elif 'same' in text_lower:
            return "Unchanged"
        else:
            return "Status unclear"
    
    def _extract_prognosis(self, text: str) -> str:
        """Extract prognosis information."""
        text_lower = text.lower()
        
        if 'full recovery' in text_lower:
            # Try to find timeframe
            match = re.search(r'within\s+(\w+\s+\w+)', text_lower)
            if match:
                return f"Full recovery expected within {match.group(1)}"
            return "Full recovery expected"
        elif 'recovery' in text_lower:
            return "Recovery expected"
        else:
            return "Prognosis not explicitly stated"
    
    def _assess_data_quality(self, entities: Dict) -> str:
        """
        Assess quality of extracted data.
        This addresses 'handling ambiguous/missing data' requirement.
        """
        total_entities = sum(len(v) for v in entities.values())
        
        if total_entities >= 10:
            return "Good - comprehensive information"
        elif total_entities >= 5:
            return "Moderate - some details missing"
        else:
            return "Limited - sparse information"


class SOAPNoteGenerator:
    """
    Generates SOAP notes from medical transcripts.
    SOAP = Subjective, Objective, Assessment, Plan
    """
    
    def __init__(self, nlp_engine: MedicalNLPEngine):
        self.nlp = nlp_engine
    
    def generate_soap_note(self, transcript: str) -> Dict:
        """Generate complete SOAP note."""
        doctor_lines, patient_lines = self.nlp.parse_conversation(transcript)
        full_text = ' '.join(doctor_lines + patient_lines)
        
        soap = {
            "subjective": self._generate_subjective(patient_lines, full_text),
            "objective": self._generate_objective(doctor_lines, full_text),
            "assessment": self._generate_assessment(full_text),
            "plan": self._generate_plan(full_text)
        }
        
        return soap
    
    def _generate_subjective(self, patient_lines: List[str], full_text: str) -> Dict:
        """Subjective section - patient's perspective."""
        patient_text = ' '.join(patient_lines).lower()
        
        # Chief complaint
        chief_complaint = "Not stated"
        if 'pain' in patient_text:
            body_parts = []
            if 'neck' in patient_text:
                body_parts.append("neck")
            if 'back' in patient_text:
                body_parts.append("back")
            if body_parts:
                chief_complaint = f"{' and '.join(body_parts).title()} pain"
        
        # History of present illness
        hpi_elements = []
        
        if 'accident' in patient_text or 'collision' in patient_text:
            hpi_elements.append("Patient involved in motor vehicle accident")
        
        # Duration
        duration_match = re.search(r'(\w+)\s+(weeks?|months?|days?)', patient_text)
        if duration_match:
            hpi_elements.append(f"Symptoms present for {duration_match.group(0)}")
        
        # Treatment history
        if 'physiotherapy' in patient_text:
            hpi_elements.append("Underwent physiotherapy treatment")
        
        # Current status
        if 'occasional' in patient_text:
            hpi_elements.append("Currently experiencing occasional symptoms")
        elif 'better' in patient_text:
            hpi_elements.append("Reports improvement in symptoms")
        
        hpi = '. '.join(hpi_elements) + '.' if hpi_elements else "History not fully documented."
        
        return {
            "chief_complaint": chief_complaint,
            "history_of_present_illness": hpi
        }
    
    def _generate_objective(self, doctor_lines: List[str], full_text: str) -> Dict:
        """Objective section - clinical findings."""
        doctor_text = ' '.join(doctor_lines).lower()
        
        # Physical exam findings
        exam_findings = []
        
        if 'range of motion' in doctor_text or 'range of movement' in doctor_text:
            exam_findings.append("Full range of motion in affected areas")
        
        if 'no tenderness' in doctor_text:
            exam_findings.append("No tenderness on palpation")
        
        if 'muscles' in doctor_text and 'spine' in doctor_text:
            exam_findings.append("Musculoskeletal examination within normal limits")
        
        if not exam_findings:
            exam_findings.append("Physical examination completed")
        
        physical_exam = '. '.join(exam_findings) + '.'
        
        # General observations
        observations = "Patient appears comfortable and in no acute distress. Normal gait and posture."
        
        return {
            "physical_exam": physical_exam,
            "observations": observations
        }
    
    def _generate_assessment(self, full_text: str) -> Dict:
        """Assessment section - diagnosis and clinical impression."""
        text_lower = full_text.lower()
        
        # Primary diagnosis
        diagnosis = "Not specified"
        if 'whiplash' in text_lower:
            diagnosis = "Whiplash injury"
        elif 'strain' in text_lower:
            diagnosis = "Musculoskeletal strain"
        
        # Severity assessment
        severity = "Mild"
        if any(term in text_lower for term in ['improving', 'better', 'occasional']):
            severity = "Mild, improving"
        elif 'severe' in text_lower or 'significant' in text_lower:
            severity = "Moderate to severe"
        
        return {
            "diagnosis": diagnosis,
            "severity": severity
        }
    
    def _generate_plan(self, full_text: str) -> Dict:
        """Plan section - treatment and follow-up."""
        text_lower = full_text.lower()
        
        # Treatment plan
        treatment_elements = []
        
        if 'physiotherapy' in text_lower:
            treatment_elements.append("Continue physiotherapy as needed")
        
        if any(term in text_lower for term in ['painkiller', 'pain relief', 'medication']):
            treatment_elements.append("Use analgesics for pain management as required")
        
        if not treatment_elements:
            treatment_elements.append("Conservative management with activity modification")
        
        treatment_plan = '. '.join(treatment_elements) + '.'
        
        # Follow-up
        follow_up = "Patient to return if symptoms worsen or persist."
        if 'six months' in text_lower or '6 months' in text_lower:
            follow_up = "Patient to return if symptoms worsen or persist beyond six months."
        
        return {
            "treatment": treatment_plan,
            "follow_up": follow_up
        }


# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(
    page_title="Medical NLP Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .result-box h4 {
        color: #1f77b4;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
    .soap-section {
        background: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .soap-section h4 {
        color: #495057;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'nlp_engine' not in st.session_state:
    with st.spinner('🔄 Loading NLP models (first time takes ~1 minute)...'):
        st.session_state.nlp_engine = MedicalNLPEngine()
        st.session_state.sentiment_analyzer = SentimentAnalyzer(
            st.session_state.nlp_engine.sentiment_pipeline
        )
        st.session_state.summarizer = MedicalSummarizer(st.session_state.nlp_engine)
        st.session_state.soap_generator = SOAPNoteGenerator(st.session_state.nlp_engine)
    st.success('✅ Models loaded!')

if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Sample transcript
SAMPLE_TRANSCRIPT = """Physician: Good morning, Ms. Jones. How are you feeling today?
Patient: Good morning, doctor. I'm doing better, but I still have some discomfort now and then.
Physician: I understand you were in a car accident last September. Can you walk me through what happened?
Patient: Yes, it was on September 1st, around 12:30 in the afternoon. I was driving from Cheadle Hulme to Manchester when I had to stop in traffic. Out of nowhere, another car hit me from behind, which pushed my car into the one in front.
Physician: That sounds like a strong impact. Were you wearing your seatbelt?
Patient: Yes, I always do.
Physician: What did you feel immediately after the accident?
Patient: At first, I was just shocked. But then I realized I had hit my head on the steering wheel, and I could feel pain in my neck and back almost right away.
Physician: Did you seek medical attention at that time?
Patient: Yes, I went to Moss Bank Accident and Emergency. They checked me over and said it was a whiplash injury, but they didn't do any X-rays. They just gave me some advice and sent me home.
Physician: How did things progress after that?
Patient: The first four weeks were rough. My neck and back pain were really bad—I had trouble sleeping and had to take painkillers regularly. It started improving after that, but I had to go through ten sessions of physiotherapy to help with the stiffness and discomfort.
Physician: That makes sense. Are you still experiencing pain now?
Patient: It's not constant, but I do get occasional backaches. It's nothing like before, though.
Physician: That's good to hear. Have you noticed any other effects, like anxiety while driving or difficulty concentrating?
Patient: No, nothing like that. I don't feel nervous driving, and I haven't had any emotional issues from the accident.
Physician: And how has this impacted your daily life? Work, hobbies, anything like that?
Patient: I had to take a week off work, but after that, I was back to my usual routine. It hasn't really stopped me from doing anything.
Physician: That's encouraging. Let's go ahead and do a physical examination to check your mobility and any lingering pain.
[Physical Examination Conducted]
Physician: Everything looks good. Your neck and back have a full range of movement, and there's no tenderness or signs of lasting damage. Your muscles and spine seem to be in good condition.
Patient: That's a relief!
Physician: Yes, your recovery so far has been quite positive. Given your progress, I'd expect you to make a full recovery within six months of the accident. There are no signs of long-term damage or degeneration.
Patient: That's great to hear. So, I don't need to worry about this affecting me in the future?
Physician: That's right. I don't foresee any long-term impact on your work or daily life. If anything changes or you experience worsening symptoms, you can always come back for a follow-up. But at this point, you're on track for a full recovery.
Patient: Thank you, doctor. I appreciate it.
Physician: You're very welcome, Ms. Jones. Take care, and don't hesitate to reach out if you need anything."""

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/stethoscope.png", width=150)
    st.markdown("### 🏥 Medical NLP Suite")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🏠 Home", "📊 NLP Summary", "😊 Sentiment Analysis", "📝 SOAP Notes", "📈 Analytics Dashboard"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("**💡 Quick Tips:**")
    st.info("""
    - Use the sample transcript to test
    - All models run locally (no API calls)
    - JSON exports available for all outputs
    """)
    
    st.markdown("---")
    st.markdown("**🎯 Project Goals:**")
    st.success("""
    ✅ Medical NER  
    ✅ Sentiment Analysis  
    ✅ SOAP Generation  
    ✅ Keyword Extraction
    """)


# ============================================================================
# PAGE: HOME
# ============================================================================

if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🏥 Medical Transcript NLP Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("### AI/ML Internship Assignment - Medical Transcription & Summarization")
    
    st.markdown("---")
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3>🔍 Medical NER</h3>
            <p>Extract symptoms, diagnoses, treatments using spaCy + custom patterns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h3>😊 Sentiment</h3>
            <p>Analyze patient emotions with DistilBERT + intent detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h3>📝 SOAP Notes</h3>
            <p>Auto-generate clinical documentation in SOAP format</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Project overview
    st.markdown("### 📖 What This App Does")
    st.write("""
    This is my solution to the physician notetaker challenge. The app processes doctor-patient 
    conversations and extracts key medical information using NLP techniques.
    
    **Technologies Used:**
    - **spaCy** (en_core_web_sm) with custom entity ruler for medical terms
    - **Transformers** (DistilBERT) for sentiment analysis
    - **TF-IDF** for keyword extraction
    - **Rule-based patterns** for diagnosis and treatment extraction
    - **Plotly** for data visualization
    """)
    
    st.markdown("### 🔧 Technical Approach")
    
    with st.expander("❓ How I handled ambiguous/missing data"):
        st.write("""
        **Challenge:** Medical transcripts often have incomplete or ambiguous information.
        
        **My Solution:**
        1. **Layered extraction**: First try spaCy NER, then fallback to regex patterns
        2. **Confidence scoring**: Mark data quality as 'Good', 'Moderate', or 'Limited'
        3. **Defaults with context**: Use "unspecified" labels when data is missing
        4. **Multiple extraction methods**: Combine NER + rule-based + keyword extraction
        
        Example: If spaCy doesn't catch "10 sessions of physiotherapy", regex catches it.
        """)
    
    with st.expander("🤖 Pre-trained models I evaluated"):
        st.write("""
        **For Medical NER:**
        - ❌ `en_core_sci_sm` - Best for medical text but large download
        - ✅ `en_core_web_sm` + custom patterns - Good balance, faster
        - ❌ BioBERT - Overkill for this demo, need more compute
        
        **For Sentiment:**
        - ✅ `distilbert-base-uncased-finetuned-sst-2-english` - Fast, decent accuracy
        - ❌ Medical-specific BERT - Couldn't find good pre-trained one
        - 💡 Future: Fine-tune on medical sentiment dataset
        
        **For Summarization:**
        - Custom rule-based (current) - Works well for structured convos
        - 💡 Future: Try BART or T5 fine-tuned on medical notes
        """)
    
    with st.expander("📚 Datasets for medical sentiment (hypothetically)"):
        st.write("""
        If I were to fine-tune BERT for medical sentiment, I'd use:
        
        1. **Medical Transcriptions** from Kaggle - Doctor-patient convos
        2. **Patient Experience Surveys** - Labeled emotional data
        3. **MIMIC-III Clinical Notes** - Real hospital data (need access)
        4. **PubMed abstracts** with sentiment labels
        5. **Health forums** (Reddit r/AskDocs) - scraped and labeled
        
        **Training approach:**
        - Start with general sentiment model (SST-2)
        - Fine-tune on medical data with labels: Anxious, Neutral, Reassured
        - Add medical vocabulary tokens
        - Use class weights to handle imbalance
        """)
    
    st.markdown("---")
    st.info("👈 Use the sidebar to navigate to different features. Start with the sample transcript!")


# ============================================================================
# PAGE: NLP SUMMARY
# ============================================================================

elif page == "📊 NLP Summary":
    st.header("📊 Medical NLP Summarization")
    st.markdown("Extract medical entities and generate structured summaries")
    
    # Input area
    col1, col2 = st.columns([3, 1])
    with col1:
        use_sample = st.checkbox("Use sample transcript", value=True)
    with col2:
        st.markdown("")  # spacing
    
    if use_sample:
        transcript = st.text_area(
            "📄 Medical Transcript",
            value=SAMPLE_TRANSCRIPT,
            height=300,
            help="Edit the sample or paste your own transcript"
        )
    else:
        transcript = st.text_area(
            "📄 Paste your transcript here",
            height=300,
            placeholder="Doctor: How are you feeling?\nPatient: I have pain in my neck..."
        )
    
    if st.button("🔍 Analyze Transcript", type="primary"):
        if not transcript.strip():
            st.error("Please enter a transcript!")
        else:
            with st.spinner("Processing with NLP models..."):
                # Generate summary
                summary = st.session_state.summarizer.generate_summary(transcript)
                st.session_state.current_summary = summary
                
                # Save to history
                st.session_state.analysis_history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "type": "Summary",
                    "patient": summary["patient_name"]
                })
            
            st.success("✅ Analysis complete!")
    
    # Display results
    if 'current_summary' in st.session_state:
        summary = st.session_state.current_summary
        
        st.markdown("---")
        st.subheader("📋 Extracted Medical Information")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("👤 Patient", summary["patient_name"])
        with col2:
            st.metric("🏥 Diagnosis", summary["diagnosis"])
        with col3:
            st.metric("📊 Status", summary["current_status"])
        with col4:
            st.metric("🎯 Data Quality", summary["data_quality"].split('-')[0].strip())
        
        # Detailed results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🤕 Symptoms Identified")
            if summary["symptoms"]:
                for symptom in summary["symptoms"]:
                    st.markdown(f"- {symptom}")
            else:
                st.info("No specific symptoms extracted")
            
            st.markdown("### 💊 Treatment History")
            if summary["treatment"]:
                for treatment in summary["treatment"]:
                    st.markdown(f"- {treatment}")
            else:
                st.info("No treatments documented")
        
        with col2:
            st.markdown("### 📈 Prognosis")
            st.info(summary["prognosis"])
            
            st.markdown("### 🔑 Key Medical Terms")
            if summary["keywords"]:
                keyword_text = ", ".join(summary["keywords"])
                st.markdown(f"```{keyword_text}```")
            else:
                st.info("No keywords extracted")
        
        # Additional extracted info
        if summary["dates"] or summary["facilities"]:
            st.markdown("### 📅 Additional Information")
            if summary["dates"]:
                st.markdown(f"**Dates:** {', '.join(summary['dates'])}")
            if summary["facilities"]:
                st.markdown(f"**Facilities:** {', '.join(summary['facilities'])}")
        
        # Export
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            json_str = json.dumps(summary, indent=2)
            st.download_button(
                "📥 Download JSON",
                json_str,
                "medical_summary.json",
                "application/json"
            )
        with col2:
            if st.button("🔄 Clear Results"):
                del st.session_state.current_summary
                st.rerun()


# ============================================================================
# PAGE: SENTIMENT ANALYSIS
# ============================================================================

elif page == "😊 Sentiment Analysis":
    st.header("😊 Patient Sentiment & Intent Analysis")
    st.markdown("Analyze emotional tone and communication intent")
    
    # Mode selection
    analysis_mode = st.radio(
        "Analysis Mode",
        ["Single Statement", "Full Transcript Analysis"],
        horizontal=True
    )
    
    if analysis_mode == "Single Statement":
        st.markdown("### 🗣️ Analyze Individual Statement")
        
        statement = st.text_area(
            "Enter patient statement",
            placeholder="e.g., I'm worried about my back pain...",
            height=100
        )
        
        if st.button("Analyze Sentiment", type="primary"):
            if statement.strip():
                with st.spinner("Analyzing..."):
                    result = st.session_state.sentiment_analyzer.analyze_sentiment(statement)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                with col1:
                    sentiment_emoji = {"Anxious": "😰", "Neutral": "😐", "Reassured": "😊"}
                    st.metric(
                        "Sentiment",
                        result["sentiment"],
                        delta=sentiment_emoji.get(result["sentiment"], "")
                    )
                with col2:
                    st.metric("Intent", result["intent"])
                with col3:
                    confidence_pct = f"{result['confidence']*100:.1f}%"
                    st.metric("Confidence", confidence_pct)
                
                # Visualization
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result['confidence'] * 100,
                    title={'text': "Confidence Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 75], 'color': "gray"},
                            {'range': [75, 100], 'color': "lightblue"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # JSON export
                st.json(result)
            else:
                st.warning("Please enter a statement to analyze")
    
    else:  # Full transcript
        st.markdown("### 📄 Analyze Complete Transcript")
        
        transcript = st.text_area(
            "Paste transcript",
            value=SAMPLE_TRANSCRIPT,
            height=250
        )
        
        if st.button("Analyze All Statements", type="primary"):
            with st.spinner("Processing all patient statements..."):
                _, patient_lines = st.session_state.nlp_engine.parse_conversation(transcript)
                
                if not patient_lines:
                    st.error("No patient statements found in transcript!")
                else:
                    results = []
                    for i, statement in enumerate(patient_lines):
                        if statement.strip():
                            result = st.session_state.sentiment_analyzer.analyze_sentiment(statement)
                            results.append({
                                "statement_num": i + 1,
                                "text": statement[:100] + "..." if len(statement) > 100 else statement,
                                **result
                            })
                    
                    st.session_state.sentiment_results = results
                    
                    # Overall statistics
                    sentiment_counts = {}
                    for r in results:
                        sent = r['sentiment']
                        sentiment_counts[sent] = sentiment_counts.get(sent, 0) + 1
                    
                    st.markdown("### 📊 Overall Sentiment Distribution")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("😰 Anxious", sentiment_counts.get("Anxious", 0))
                    with col2:
                        st.metric("😐 Neutral", sentiment_counts.get("Neutral", 0))
                    with col3:
                        st.metric("😊 Reassured", sentiment_counts.get("Reassured", 0))
                    
                    # Pie chart
                    if sentiment_counts:
                        fig = px.pie(
                            values=list(sentiment_counts.values()),
                            names=list(sentiment_counts.keys()),
                            title="Sentiment Distribution",
                            color_discrete_map={
                                "Anxious": "#ff6b6b",
                                "Neutral": "#95a5a6",
                                "Reassured": "#51cf66"
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed results
                    st.markdown("### 📝 Individual Statements")
                    for i, result in enumerate(results):
                        with st.expander(f"Statement {i+1}: {result['text'][:50]}..."):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"**Sentiment:** {result['sentiment']}")
                            with col2:
                                st.write(f"**Intent:** {result['intent']}")
                            with col3:
                                st.write(f"**Confidence:** {result['confidence']:.1%}")
                            st.markdown(f"*Full text:* {result['text']}")
                    
                    # Export
                    json_str = json.dumps(results, indent=2)
                    st.download_button(
                        "📥 Download Sentiment Analysis",
                        json_str,
                        "sentiment_analysis.json",
                        "application/json"
                    )


# ============================================================================
# PAGE: SOAP NOTES
# ============================================================================

elif page == "📝 SOAP Notes":
    st.header("📝 SOAP Note Generator")
    st.markdown("Automatically generate clinical documentation in SOAP format")
    
    # Info box
    with st.expander("ℹ️ What is a SOAP Note?"):
        st.write("""
        **SOAP** is a structured method for documenting patient encounters:
        
        - **S**ubjective: Patient's symptoms and history (what they tell you)
        - **O**bjective: Clinical findings from examination (what you observe)
        - **A**ssessment: Diagnosis and clinical judgment
        - **P**lan: Treatment plan and follow-up
        
        This format is standard in medical documentation.
        """)
    
    transcript = st.text_area(
        "📄 Medical Transcript",
        value=SAMPLE_TRANSCRIPT,
        height=250
    )
    
    if st.button("📝 Generate SOAP Note", type="primary"):
        if not transcript.strip():
            st.error("Please enter a transcript!")
        else:
            with st.spinner("Generating SOAP note..."):
                soap_note = st.session_state.soap_generator.generate_soap_note(transcript)
                st.session_state.current_soap = soap_note
                
                # Save to history
                st.session_state.analysis_history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "type": "SOAP",
                    "patient": "N/A"
                })
            
            st.success("✅ SOAP note generated!")
    
    # Display SOAP note
    if 'current_soap' in st.session_state:
        soap = st.session_state.current_soap
        
        st.markdown("---")
        
        # Subjective
        st.markdown("""
        <div class='soap-section'>
            <h4>📋 SUBJECTIVE</h4>
        </div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("**Chief Complaint:**")
        with col2:
            st.write(soap['subjective']['chief_complaint'])
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("**History:**")
        with col2:
            st.write(soap['subjective']['history_of_present_illness'])
        
        # Objective
        st.markdown("""
        <div class='soap-section'>
            <h4>🔬 OBJECTIVE</h4>
        </div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("**Physical Exam:**")
        with col2:
            st.write(soap['objective']['physical_exam'])
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("**Observations:**")
        with col2:
            st.write(soap['objective']['observations'])
        
        # Assessment
        st.markdown("""
        <div class='soap-section'>
            <h4>🎯 ASSESSMENT</h4>
        </div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("**Diagnosis:**")
        with col2:
            st.write(soap['assessment']['diagnosis'])
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("**Severity:**")
        with col2:
            st.write(soap['assessment']['severity'])
        
        # Plan
        st.markdown("""
        <div class='soap-section'>
            <h4>📋 PLAN</h4>
        </div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("**Treatment:**")
        with col2:
            st.write(soap['plan']['treatment'])
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("**Follow-up:**")
        with col2:
            st.write(soap['plan']['follow_up'])
        
        # Export options
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            json_str = json.dumps(soap, indent=2)
            st.download_button(
                "📥 Download as JSON",
                json_str,
                "soap_note.json",
                "application/json"
            )
        with col2:
            # Create formatted text version
            text_version = f"""SOAP NOTE
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

SUBJECTIVE
Chief Complaint: {soap['subjective']['chief_complaint']}
History of Present Illness: {soap['subjective']['history_of_present_illness']}

OBJECTIVE
Physical Examination: {soap['objective']['physical_exam']}
Observations: {soap['objective']['observations']}

ASSESSMENT
Diagnosis: {soap['assessment']['diagnosis']}
Severity: {soap['assessment']['severity']}

PLAN
Treatment: {soap['plan']['treatment']}
Follow-up: {soap['plan']['follow_up']}
"""
            st.download_button(
                "📄 Download as TXT",
                text_version,
                "soap_note.txt",
                "text/plain"
            )


# ============================================================================
# PAGE: ANALYTICS DASHBOARD
# ============================================================================

elif page == "📈 Analytics Dashboard":
    st.header("📈 Analytics Dashboard")
    st.markdown("Overview of all analyses performed in this session")
    
    if not st.session_state.analysis_history:
        st.info("👋 No analyses yet! Run some analyses to see stats here.")
        st.markdown("### 🎯 Get Started")
        st.write("Try these features:")
        st.write("- 📊 Generate a medical summary")
        st.write("- 😊 Analyze patient sentiment")
        st.write("- 📝 Create a SOAP note")
    else:
        # Summary stats
        total_analyses = len(st.session_state.analysis_history)
        st.metric("📊 Total Analyses", total_analyses)
        
        # Analysis history table
        st.markdown("### 📋 Analysis History")
        df = pd.DataFrame(st.session_state.analysis_history)
        st.dataframe(df, use_container_width=True)
        
        # Visualization
        st.markdown("### 📊 Analysis Type Distribution")
        type_counts = df['type'].value_counts()
        fig = px.bar(
            x=type_counts.index,
            y=type_counts.values,
            labels={'x': 'Analysis Type', 'y': 'Count'},
            title="Types of Analyses Performed"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Timeline
        st.markdown("### ⏰ Analysis Timeline")
        fig2 = px.scatter(
            df,
            x='timestamp',
            y='type',
            color='type',
            title="Analysis Activity Over Time"
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.analysis_history = []
            st.rerun()


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p>🎓 <b>Medical Transcript NLP Analyzer</b> | AI/ML Internship Project</p>
    <p>Built with: spaCy • Transformers • Streamlit • Plotly</p>
    <p><i>⚠️ Demo purposes only - not for clinical use</i></p>
</div>
""", unsafe_allow_html=True)
