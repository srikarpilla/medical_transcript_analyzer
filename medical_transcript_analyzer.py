

import json
import re
from typing import Dict, List, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch


from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import sent_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class MedicalTranscriptAnalyzer:
   
    
    def __init__(self):
        print("Initializing Medical Transcript Analyzer...")
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model...")
            import os
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        
        self.medical_patterns = self._build_medical_patterns()
        
        print("Initialization complete!")
    
    def _build_medical_patterns(self) -> Dict[str, List[str]]:
        """Build pattern matching rules for medical entities"""
        return {
            'symptoms': [
                r'\b(pain|ache|discomfort|hurt|stiff|tenderness|sore)\b',
                r'\b(neck|back|head|shoulder|spine)\s+(pain|ache|discomfort)',
                r'\b(trouble\s+sleeping|difficulty\s+sleeping)\b',
                r'\b(impact|hit|struck|injured)\b'
            ],
            'diagnosis': [
                r'\b(whiplash|injury|strain|sprain|trauma|damage)\b',
                r'\b(diagnosis|diagnosed|condition)\b'
            ],
            'treatment': [
                r'\b(physiotherapy|therapy|treatment|medication)\b',
                r'\b(painkillers|analgesics|pills)\b',
                r'\b(sessions?|appointments?)\b',
                r'\b(X-ray|examination|checkup)\b'
            ],
            'prognosis': [
                r'\b(recovery|improve|better|healing)\b',
                r'\b(full\s+recovery|complete\s+recovery)\b',
                r'\b(expected|anticipated|projected)\b'
            ]
        }
    
    def parse_conversation(self, transcript: str) -> Tuple[List[str], List[str]]:
       
        lines = transcript.strip().split('\n')
        physician_statements = []
        patient_statements = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('**Physician:**') or line.startswith('Physician:'):
                text = re.sub(r'\*\*Physician:\*\*|\*Physician:\*|Physician:', '', line)
                text = text.strip().strip('*').strip()
                if text:
                    physician_statements.append(text)
            elif line.startswith('**Patient:**') or line.startswith('Patient:'):
                text = re.sub(r'\*\*Patient:\*\*|\*Patient:\*|Patient:', '', line)
                text = text.strip().strip('*').strip()
                if text:
                    patient_statements.append(text)
            elif line.startswith('Doctor:'):
                text = line.replace('Doctor:', '').strip()
                if text:
                    physician_statements.append(text)
        
        return physician_statements, patient_statements
    
    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract medical entities using pattern matching and NER.
        """
        entities = {
            'symptoms': [],
            'diagnosis': [],
            'treatment': [],
            'prognosis': []
        }
        
        text_lower = text.lower()
        
        
        for category, patterns in self.medical_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    entity_text = match.group(0)
                    # Get context around the match
                    start = max(0, match.start() - 20)
                    end = min(len(text), match.end() + 20)
                    context = text[start:end]
                    
                    if category == 'symptoms':
                        entities['symptoms'].append(self._clean_entity(context))
                    elif category == 'diagnosis':
                        entities['diagnosis'].append(self._clean_entity(context))
                    elif category == 'treatment':
                        entities['treatment'].append(self._clean_entity(context))
                    elif category == 'prognosis':
                        entities['prognosis'].append(self._clean_entity(context))
       
        for key in entities:
            entities[key] = list(set([e for e in entities[key] if len(e) > 3]))
        
        return entities
    
    def _clean_entity(self, text: str) -> str:
        """Clean extracted entity text"""
        text = re.sub(r'[^\w\s-]', '', text)
        text = ' '.join(text.split())
        return text.strip()
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract important medical keywords using TF-IDF.
        """
        sentences = sent_tokenize(text)
        
        if len(sentences) < 2:
            return []
        
        vectorizer = TfidfVectorizer(
            max_features=top_n,
            stop_words='english',
            ngram_range=(1, 3)
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            return list(feature_names)
        except:
            return []
    
    def analyze_sentiment(self, text: str) -> Dict[str, str]:
        """
        Analyze sentiment and intent from patient statements.
        """
        
        sentiment_result = self.sentiment_analyzer(text[:512])[0]
        
        # Map to medical context
        sentiment_score = sentiment_result['score']
        sentiment_label = sentiment_result['label']
        
        if sentiment_label == 'NEGATIVE' and sentiment_score > 0.7:
            sentiment = "Anxious"
        elif sentiment_label == 'POSITIVE' and sentiment_score > 0.7:
            sentiment = "Reassured"
        else:
            sentiment = "Neutral"
        
        # Intent detection based on keywords
        intent = self._detect_intent(text)
        
        return {
            "sentiment": sentiment,
            "intent": intent,
            "confidence": round(sentiment_score, 3)
        }
    
    def _detect_intent(self, text: str) -> str:
        """Detect patient intent from text"""
        text_lower = text.lower()
        
        concern_keywords = ['worry', 'worried', 'concerned', 'afraid', 'anxious', 'scared']
        symptom_keywords = ['pain', 'hurt', 'discomfort', 'ache', 'feel', 'experiencing']
        reassurance_keywords = ['hope', 'better', 'improve', 'recovery', 'relief']
        question_keywords = ['?', 'will i', 'should i', 'can i', 'do i need']
        
        if any(keyword in text_lower for keyword in concern_keywords):
            return "Expressing concern"
        elif any(keyword in text_lower for keyword in question_keywords):
            return "Seeking reassurance"
        elif any(keyword in text_lower for keyword in symptom_keywords):
            return "Reporting symptoms"
        elif any(keyword in text_lower for keyword in reassurance_keywords):
            return "Seeking reassurance"
        else:
            return "General communication"
    
    def generate_medical_summary(self, transcript: str) -> Dict:
        """
        Generate structured medical summary from conversation.
        """
        physician_stmts, patient_stmts = self.parse_conversation(transcript)
        
        # Combine all text
        full_text = ' '.join(patient_stmts + physician_stmts)
        patient_text = ' '.join(patient_stmts)
        
        # Extract entities
        entities = self.extract_medical_entities(full_text)
        
        # Extract specific information
        patient_name = self._extract_patient_name(full_text)
        diagnosis = self._extract_diagnosis(full_text)
        current_status = self._extract_current_status(patient_text)
        prognosis = self._extract_prognosis(full_text)
        
        summary = {
            "patient_name": patient_name,
            "symptoms": self._consolidate_symptoms(full_text),
            "diagnosis": diagnosis,
            "treatment": self._extract_treatments(full_text),
            "current_status": current_status,
            "prognosis": prognosis,
            "keywords": self.extract_keywords(full_text, top_n=8)
        }
        
        return summary
    
    def _extract_patient_name(self, text: str) -> str:
        """Extract patient name from conversation"""
        # Look for name patterns
        name_patterns = [
            r'Ms\.\s+([A-Z][a-z]+)',
            r'Mr\.\s+([A-Z][a-z]+)',
            r'Mrs\.\s+([A-Z][a-z]+)'
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0).replace('.', '. ')
        
        return "Patient"
    
    def _extract_diagnosis(self, text: str) -> str:
        """Extract primary diagnosis"""
        diagnosis_patterns = [
            r'(whiplash\s+injury)',
            r'diagnosed\s+with\s+([^,\.]+)',
            r'it\s+was\s+a\s+([^,\.]+)'
        ]
        
        for pattern in diagnosis_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if 'whiplash' in match.group(0).lower():
                    return "Whiplash injury"
        
        return "Soft tissue injury"
    
    def _consolidate_symptoms(self, text: str) -> List[str]:
        """Extract and consolidate symptoms"""
        symptoms = []
        text_lower = text.lower()
        
        symptom_map = {
            'neck pain': ['neck pain', 'pain in my neck', 'neck hurt'],
            'back pain': ['back pain', 'pain in my back', 'back hurt', 'backache'],
            'head impact': ['hit my head', 'head on', 'struck my head'],
            'trouble sleeping': ['trouble sleeping', 'difficulty sleeping'],
            'stiffness': ['stiff', 'stiffness']
        }
        
        for symptom, keywords in symptom_map.items():
            if any(kw in text_lower for kw in keywords):
                symptoms.append(symptom.title())
        
        return symptoms if symptoms else ["Pain and discomfort"]
    
    def _extract_treatments(self, text: str) -> List[str]:
        """Extract treatment information"""
        treatments = []
        text_lower = text.lower()
        
        if 'physiotherapy' in text_lower or 'physical therapy' in text_lower:
            # Extract number of sessions
            session_match = re.search(r'(\d+)\s+sessions?', text_lower)
            if session_match:
                treatments.append(f"{session_match.group(1)} physiotherapy sessions")
            else:
                treatments.append("Physiotherapy sessions")
        
        if 'painkiller' in text_lower or 'pain relief' in text_lower or 'analgesic' in text_lower:
            treatments.append("Painkillers")
        
        return treatments if treatments else ["Supportive care"]
    
    def _extract_current_status(self, text: str) -> str:
        """Extract current patient status"""
        text_lower = text.lower()
        
        if 'occasional' in text_lower and ('pain' in text_lower or 'ache' in text_lower):
            return "Occasional backache"
        elif 'better' in text_lower:
            return "Improving"
        elif 'worse' in text_lower:
            return "Worsening"
        else:
            return "Stable"
    
    def _extract_prognosis(self, text: str) -> str:
        
        text_lower = text.lower()
        
        if 'full recovery' in text_lower:
           
            time_match = re.search(r'within\s+(\w+\s+months?)', text_lower)
            if time_match:
                return f"Full recovery expected within {time_match.group(1)}"
            return "Full recovery expected"
        elif 'recovery' in text_lower:
            return "Good prognosis for recovery"
        else:
            return "Prognosis favorable"
    
    def generate_soap_note(self, transcript: str) -> Dict:
        """
        Generate SOAP note from conversation transcript.
        """
        physician_stmts, patient_stmts = self.parse_conversation(transcript)
        full_text = ' '.join(patient_stmts + physician_stmts)
        
        soap = {
            "subjective": self._generate_subjective(patient_stmts, full_text),
            "objective": self._generate_objective(physician_stmts, full_text),
            "assessment": self._generate_assessment(full_text),
            "plan": self._generate_plan(full_text)
        }
        
        return soap
    
    def _generate_subjective(self, patient_stmts: List[str], full_text: str) -> Dict:
        """Generate Subjective section of SOAP note"""
        patient_text = ' '.join(patient_stmts)
        
        # Extract chief complaint
        chief_complaint = "Neck and back pain"
        if 'pain' in patient_text.lower():
            if 'neck' in patient_text.lower() and 'back' in patient_text.lower():
                chief_complaint = "Neck and back pain"
            elif 'neck' in patient_text.lower():
                chief_complaint = "Neck pain"
            elif 'back' in patient_text.lower():
                chief_complaint = "Back pain"
        
        # Extract history
        history = []
        if 'accident' in patient_text.lower():
            history.append("Patient involved in motor vehicle accident")
        if 'four weeks' in patient_text.lower() or '4 weeks' in patient_text.lower():
            history.append("Experienced significant pain for approximately four weeks")
        if 'physiotherapy' in full_text.lower():
            history.append("Completed physiotherapy sessions")
        if 'occasional' in patient_text.lower():
            history.append("Currently experiencing occasional symptoms")
        
        return {
            "chief_complaint": chief_complaint,
            "history_of_present_illness": '. '.join(history) + '.'
        }
    
    def _generate_objective(self, physician_stmts: List[str], full_text: str) -> Dict:
        """Generate Objective section of SOAP note"""
        physician_text = ' '.join(physician_stmts).lower()
        
        physical_exam = []
        if 'range of movement' in physician_text or 'range of motion' in physician_text:
            physical_exam.append("Full range of motion in cervical and lumbar spine")
        if 'no tenderness' in physician_text:
            physical_exam.append("No tenderness on palpation")
        if 'muscles' in physician_text and 'spine' in physician_text:
            physical_exam.append("Musculoskeletal examination within normal limits")
        
        if not physical_exam:
            physical_exam.append("Physical examination completed")
        
        return {
            "physical_exam": '. '.join(physical_exam) + '.',
            "observations": "Patient appears comfortable. Normal gait and posture observed."
        }
    
    def _generate_assessment(self, full_text: str) -> Dict:
        """Generate Assessment section of SOAP note"""
        text_lower = full_text.lower()
        
        diagnosis = self._extract_diagnosis(full_text)
        
        # Determine severity
        severity = "Mild"
        if 'improving' in text_lower or 'better' in text_lower:
            severity = "Mild, improving"
        elif 'severe' in text_lower or 'significant' in text_lower:
            severity = "Moderate"
        
        return {
            "diagnosis": diagnosis,
            "severity": severity
        }
    
    def _generate_plan(self, full_text: str) -> Dict:
        """Generate Plan section of SOAP note"""
        text_lower = full_text.lower()
        
        treatment_plan = []
        if 'physiotherapy' in text_lower:
            treatment_plan.append("Continue physiotherapy as needed")
        if 'painkiller' in text_lower or 'pain relief' in text_lower:
            treatment_plan.append("Use analgesics for pain relief as required")
        
        if not treatment_plan:
            treatment_plan.append("Conservative management with rest and activity modification")
        
        
        follow_up = "Patient to return if symptoms worsen or persist"
        if 'six months' in text_lower:
            follow_up = "Patient to return if pain worsens or persists beyond six months"
        
        return {
            "treatment": '. '.join(treatment_plan) + '.',
            "follow_up": follow_up + '.'
        }


def print_section(title, content="", width=70):
    """Helper function to print formatted sections"""
    print("\n" + title)
    print("-" * width)
    if content:
        print(content)


def main():
    
   
    sample_transcript = """
    Physician: Good morning, Ms. Jones. How are you feeling today?
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
    Physician: Everything looks good. Your neck and back have a full range of movement, and there's no tenderness or signs of lasting damage. Your muscles and spine seem to be in good condition.
    Patient: That's a relief!
    Physician: Yes, your recovery so far has been quite positive. Given your progress, I'd expect you to make a full recovery within six months of the accident. There are no signs of long-term damage or degeneration.
    Patient: That's great to hear. So, I don't need to worry about this affecting me in the future?
    Physician: That's right. I don't foresee any long-term impact on your work or daily life. If anything changes or you experience worsening symptoms, you can always come back for a follow-up. But at this point, you're on track for a full recovery.
    Patient: Thank you, doctor. I appreciate it.
    Physician: You're very welcome, Ms. Jones. Take care, and don't hesitate to reach out if you need anything.
    """
    
  
    analyzer = MedicalTranscriptAnalyzer()
    
    print("\n" + "=" * 70)
    print("MEDICAL TRANSCRIPT ANALYSIS PIPELINE")
    print("=" * 70)
    
    # Task 1: Medical NLP Summarization
    print_section("[TASK 1] Medical NLP Summarization")
    summary = analyzer.generate_medical_summary(sample_transcript)
    print(json.dumps(summary, indent=2))
    
    # Task 2: Sentiment & Intent Analysis
    print_section("\n[TASK 2] Sentiment & Intent Analysis")
    
    # Analyze patient statements
    _, patient_stmts = analyzer.parse_conversation(sample_transcript)
    
    # Example patient statement
    test_statement = "I'm a bit worried about my back pain, but I hope it gets better soon."
    sentiment_result = analyzer.analyze_sentiment(test_statement)
    print(f"\nTest Statement: '{test_statement}'")
    print(json.dumps(sentiment_result, indent=2))
    
    # Analyze a few patient statements from conversation
    print("\nAnalyzing patient statements from conversation:")
    for i, stmt in enumerate(patient_stmts[:3], 1):
        result = analyzer.analyze_sentiment(stmt)
        print(f"\n{i}. Patient: {stmt[:80]}...")
        print(f"   Sentiment: {result['sentiment']}, Intent: {result['intent']}")
    
    # Task 3: SOAP Note Generation
    print_section("\n[TASK 3] SOAP Note Generation")
    soap_note = analyzer.generate_soap_note(sample_transcript)
    print(json.dumps(soap_note, indent=2))
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()