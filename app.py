"""
Medical Transcription Analysis Web Application
Built with Streamlit for interactive medical NLP analysis
"""

import streamlit as st
import json
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Import the analyzer class
from medical_transcript_analyzer import MedicalTranscriptAnalyzer

# Page configuration
st.set_page_config(
    page_title="Medical Transcript Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.8rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    h1 {
        color: #1f77b4;
        font-weight: 700;
    }
    h2 {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .json-output {
        background-color: #282c34;
        color: #abb2bf;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: 'Courier New', monospace;
        overflow-x: auto;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzer' not in st.session_state:
    with st.spinner('🔄 Initializing Medical NLP Pipeline...'):
        st.session_state.analyzer = MedicalTranscriptAnalyzer()
    st.success('✅ System Ready!')

if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Sample conversation for demo
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
Physician: Everything looks good. Your neck and back have a full range of movement, and there's no tenderness or signs of lasting damage. Your muscles and spine seem to be in good condition.
Patient: That's a relief!
Physician: Yes, your recovery so far has been quite positive. Given your progress, I'd expect you to make a full recovery within six months of the accident. There are no signs of long-term damage or degeneration.
Patient: That's great to hear. So, I don't need to worry about this affecting me in the future?
Physician: That's right. I don't foresee any long-term impact on your work or daily life. If anything changes or you experience worsening symptoms, you can always come back for a follow-up. But at this point, you're on track for a full recovery.
Patient: Thank you, doctor. I appreciate it.
Physician: You're very welcome, Ms. Jones. Take care, and don't hesitate to reach out if you need anything."""

# Header
st.title("🏥 Medical Transcript Analysis System")
st.markdown("### AI-Powered Clinical Documentation & Analysis Platform")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/medical-doctor.png", width=150)
    st.title("Navigation")
    
    page = st.radio(
        "Select Analysis Module:",
        ["🏠 Home", "📝 Medical Summarization", "😊 Sentiment Analysis", "📋 SOAP Note Generator", "📊 Analytics Dashboard"],
        index=0
    )
    
    st.markdown("---")
    st.subheader("ℹ️ About")
    st.info(
        """
        This system uses advanced NLP to:
        - Extract medical entities
        - Analyze patient sentiment
        - Generate SOAP notes
        - Provide clinical insights
        """
    )
    
    st.markdown("---")
    st.caption("🔒 Demo Version - Not for Clinical Use")
    st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d')}")

# Main content area
if page == "🏠 Home":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <h3>🎯 Medical NER</h3>
                <p>Extract symptoms, diagnoses, and treatments</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card">
                <h3>😊 Sentiment AI</h3>
                <p>Analyze patient emotional state</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="metric-card">
                <h3>📋 SOAP Notes</h3>
                <p>Auto-generate clinical documentation</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.header("🚀 Quick Start Guide")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### How to Use This System
        
        1. **Select a Module** from the sidebar
        2. **Input or Load** a physician-patient conversation
        3. **Click Analyze** to process the transcript
        4. **Review Results** in structured format
        5. **Export** analysis as JSON or PDF
        
        ### Features
        
        - ✅ Named Entity Recognition (NER)
        - ✅ Keyword Extraction
        - ✅ Sentiment Classification
        - ✅ Intent Detection
        - ✅ SOAP Note Generation
        - ✅ Clinical Insights Dashboard
        """)
    
    with col2:
        st.info("💡 **Tip**: Use the sample transcript to test the system quickly!")
        
        if st.button("🎬 Try Sample Analysis", type="primary"):
            st.session_state.sample_loaded = True
            st.success("Sample loaded! Navigate to any analysis module to begin.")
    
    st.markdown("---")
    
    # Recent Analysis History
    st.subheader("📊 Recent Analysis History")
    if st.session_state.analysis_history:
        history_df = pd.DataFrame(st.session_state.analysis_history)
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No analysis history yet. Start by analyzing a transcript!")

elif page == "📝 Medical Summarization":
    st.header("📝 Medical NLP Summarization")
    st.markdown("Extract medical entities and generate structured summaries from physician-patient conversations.")
    
    # Input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Input Transcript")
    
    with col2:
        if st.button("📥 Load Sample", type="secondary"):
            st.session_state.sample_loaded = True
    
    # Text input
    if 'sample_loaded' in st.session_state and st.session_state.sample_loaded:
        transcript = st.text_area(
            "Enter physician-patient conversation:",
            value=SAMPLE_TRANSCRIPT,
            height=300,
            help="Format: 'Physician: ... Patient: ...'"
        )
    else:
        transcript = st.text_area(
            "Enter physician-patient conversation:",
            height=300,
            placeholder="Physician: How are you feeling today?\nPatient: I've been experiencing...",
            help="Format: 'Physician: ... Patient: ...'"
        )
    
    # Analysis button
    if st.button("🔍 Analyze Transcript", type="primary", use_container_width=True):
        if transcript.strip():
            with st.spinner("🔄 Processing transcript..."):
                # Generate summary
                summary = st.session_state.analyzer.generate_medical_summary(transcript)
                
                # Store in session state
                st.session_state.current_summary = summary
                st.session_state.current_transcript = transcript
                
                # Add to history
                st.session_state.analysis_history.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'type': 'Medical Summary',
                    'patient': summary.get('patient_name', 'N/A')
                })
                
            st.success("✅ Analysis Complete!")
        else:
            st.warning("⚠️ Please enter a transcript first.")
    
    # Display results
    if 'current_summary' in st.session_state:
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        summary = st.session_state.current_summary
        
        with col1:
            st.metric("👤 Patient", summary.get('patient_name', 'N/A'))
        with col2:
            st.metric("🩺 Diagnosis", summary.get('diagnosis', 'N/A'))
        with col3:
            st.metric("💊 Treatments", len(summary.get('treatment', [])))
        with col4:
            st.metric("🔑 Keywords", len(summary.get('keywords', [])))
        
        # Detailed sections
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "🎯 Entities", "🔑 Keywords", "💾 Export"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🩺 Clinical Information")
                st.markdown(f"**Diagnosis:** {summary.get('diagnosis', 'N/A')}")
                st.markdown(f"**Current Status:** {summary.get('current_status', 'N/A')}")
                st.markdown(f"**Prognosis:** {summary.get('prognosis', 'N/A')}")
            
            with col2:
                st.markdown("#### 📊 Symptoms & Treatment")
                
                if summary.get('symptoms'):
                    st.markdown("**Symptoms:**")
                    for symptom in summary.get('symptoms', []):
                        st.markdown(f"- {symptom}")
                
                if summary.get('treatment'):
                    st.markdown("**Treatment:**")
                    for treatment in summary.get('treatment', []):
                        st.markdown(f"- {treatment}")
        
        with tab2:
            st.markdown("#### 🎯 Extracted Medical Entities")
            
            # Create visualization
            entity_data = {
                'Category': [],
                'Count': []
            }
            
            if summary.get('symptoms'):
                entity_data['Category'].append('Symptoms')
                entity_data['Count'].append(len(summary['symptoms']))
            
            if summary.get('treatment'):
                entity_data['Category'].append('Treatments')
                entity_data['Count'].append(len(summary['treatment']))
            
            if entity_data['Category']:
                fig = px.bar(
                    entity_data,
                    x='Category',
                    y='Count',
                    title='Medical Entity Distribution',
                    color='Category',
                    color_discrete_sequence=['#667eea', '#764ba2']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("#### 🔑 Important Medical Keywords")
            
            keywords = summary.get('keywords', [])
            if keywords:
                # Create word cloud-style display
                cols = st.columns(3)
                for idx, keyword in enumerate(keywords):
                    with cols[idx % 3]:
                        st.markdown(f"""
                            <div style="background-color: #e3f2fd; padding: 0.5rem; 
                                 border-radius: 0.5rem; margin: 0.2rem; text-align: center;">
                                <b>{keyword}</b>
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No keywords extracted.")
        
        with tab4:
            st.markdown("#### 💾 Export Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                json_output = json.dumps(summary, indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_output,
                    file_name=f"medical_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col2:
                # Format as readable text
                text_output = f"""MEDICAL SUMMARY REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Patient: {summary.get('patient_name', 'N/A')}
Diagnosis: {summary.get('diagnosis', 'N/A')}
Current Status: {summary.get('current_status', 'N/A')}
Prognosis: {summary.get('prognosis', 'N/A')}

Symptoms:
{chr(10).join('- ' + s for s in summary.get('symptoms', []))}

Treatment:
{chr(10).join('- ' + t for t in summary.get('treatment', []))}

Keywords:
{', '.join(summary.get('keywords', []))}
"""
                st.download_button(
                    label="📄 Download Text",
                    data=text_output,
                    file_name=f"medical_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            # JSON preview
            st.markdown("**JSON Preview:**")
            st.code(json_output, language='json')

elif page == "😊 Sentiment Analysis":
    st.header("😊 Patient Sentiment & Intent Analysis")
    st.markdown("Analyze patient emotional state and communication intent using AI.")
    
    # Input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Input Options")
    
    with col2:
        input_mode = st.selectbox("Mode", ["Single Statement", "Full Transcript"])
    
    if input_mode == "Single Statement":
        # Single statement analysis
        patient_statement = st.text_area(
            "Enter patient statement:",
            placeholder="e.g., I'm worried about my back pain, but I hope it gets better soon.",
            height=100
        )
        
        if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
            if patient_statement.strip():
                with st.spinner("🔄 Analyzing sentiment..."):
                    result = st.session_state.analyzer.analyze_sentiment(patient_statement)
                    st.session_state.sentiment_result = result
                    st.session_state.analyzed_statement = patient_statement
                
                st.success("✅ Analysis Complete!")
            else:
                st.warning("⚠️ Please enter a statement first.")
        
        # Display result
        if 'sentiment_result' in st.session_state:
            st.markdown("---")
            result = st.session_state.sentiment_result
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sentiment_color = {
                    'Anxious': '🔴',
                    'Neutral': '🟡',
                    'Reassured': '🟢'
                }
                st.metric(
                    "Sentiment",
                    f"{sentiment_color.get(result['sentiment'], '⚪')} {result['sentiment']}"
                )
            
            with col2:
                st.metric("Intent", result['intent'])
            
            with col3:
                st.metric("Confidence", f"{result['confidence']:.1%}")
            
            # Visualization
            st.markdown("#### 📊 Sentiment Breakdown")
            
            sentiment_scores = {
                'Anxious': 0.8 if result['sentiment'] == 'Anxious' else 0.2,
                'Neutral': 0.8 if result['sentiment'] == 'Neutral' else 0.2,
                'Reassured': 0.8 if result['sentiment'] == 'Reassured' else 0.2
            }
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(sentiment_scores.keys()),
                    y=list(sentiment_scores.values()),
                    marker_color=['#ef5350', '#ffa726', '#66bb6a'],
                    text=[f"{v:.0%}" for v in sentiment_scores.values()],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Sentiment Classification Scores",
                xaxis_title="Sentiment Category",
                yaxis_title="Probability",
                yaxis_range=[0, 1],
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Export
            st.markdown("---")
            json_output = json.dumps(result, indent=2)
            st.download_button(
                label="📥 Download Result (JSON)",
                data=json_output,
                file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    else:
        # Full transcript analysis
        if 'sample_loaded' in st.session_state and st.session_state.sample_loaded:
            transcript = st.text_area(
                "Enter full conversation transcript:",
                value=SAMPLE_TRANSCRIPT,
                height=300
            )
        else:
            transcript = st.text_area(
                "Enter full conversation transcript:",
                height=300,
                placeholder="Physician: ... Patient: ..."
            )
        
        if st.button("🔍 Analyze All Patient Statements", type="primary", use_container_width=True):
            if transcript.strip():
                with st.spinner("🔄 Analyzing all patient statements..."):
                    _, patient_stmts = st.session_state.analyzer.parse_conversation(transcript)
                    
                    results = []
                    for stmt in patient_stmts:
                        if stmt.strip():
                            result = st.session_state.analyzer.analyze_sentiment(stmt)
                            results.append({
                                'statement': stmt[:100] + '...' if len(stmt) > 100 else stmt,
                                'sentiment': result['sentiment'],
                                'intent': result['intent'],
                                'confidence': result['confidence']
                            })
                    
                    st.session_state.sentiment_results = results
                
                st.success(f"✅ Analyzed {len(results)} patient statements!")
            else:
                st.warning("⚠️ Please enter a transcript first.")
        
        # Display results
        if 'sentiment_results' in st.session_state:
            st.markdown("---")
            st.subheader("📊 Sentiment Analysis Results")
            
            results = st.session_state.sentiment_results
            
            # Overview metrics
            sentiment_counts = {}
            for r in results:
                sentiment_counts[r['sentiment']] = sentiment_counts.get(r['sentiment'], 0) + 1
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Statements", len(results))
            with col2:
                st.metric("😰 Anxious", sentiment_counts.get('Anxious', 0))
            with col3:
                st.metric("😐 Neutral", sentiment_counts.get('Neutral', 0))
            with col4:
                st.metric("😊 Reassured", sentiment_counts.get('Reassured', 0))
            
            # Detailed results
            st.markdown("#### Detailed Analysis")
            
            for idx, result in enumerate(results, 1):
                with st.expander(f"Statement {idx}: {result['statement'][:80]}..."):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"**Sentiment:** {result['sentiment']}")
                    with col2:
                        st.markdown(f"**Intent:** {result['intent']}")
                    with col3:
                        st.markdown(f"**Confidence:** {result['confidence']:.1%}")
                    
                    st.markdown(f"**Full Statement:** {result['statement']}")
            
            # Pie chart
            st.markdown("#### Sentiment Distribution")
            
            fig = px.pie(
                values=list(sentiment_counts.values()),
                names=list(sentiment_counts.keys()),
                color=list(sentiment_counts.keys()),
                color_discrete_map={'Anxious': '#ef5350', 'Neutral': '#ffa726', 'Reassured': '#66bb6a'}
            )
            
            st.plotly_chart(fig, use_container_width=True)

elif page == "📋 SOAP Note Generator":
    st.header("📋 SOAP Note Generator")
    st.markdown("Automatically generate structured SOAP notes from physician-patient conversations.")
    
    # Info box
    st.info("""
    **SOAP Note Format:**
    - **S**ubjective: Patient's reported symptoms and history
    - **O**bjective: Observable and measurable data
    - **A**ssessment: Diagnosis and clinical impression
    - **P**lan: Treatment plan and follow-up
    """)
    
    # Input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Input Transcript")
    
    with col2:
        if st.button("📥 Load Sample", type="secondary"):
            st.session_state.sample_loaded = True
    
    if 'sample_loaded' in st.session_state and st.session_state.sample_loaded:
        transcript = st.text_area(
            "Enter physician-patient conversation:",
            value=SAMPLE_TRANSCRIPT,
            height=300
        )
    else:
        transcript = st.text_area(
            "Enter physician-patient conversation:",
            height=300,
            placeholder="Physician: ... Patient: ..."
        )
    
    if st.button("📝 Generate SOAP Note", type="primary", use_container_width=True):
        if transcript.strip():
            with st.spinner("🔄 Generating SOAP note..."):
                soap_note = st.session_state.analyzer.generate_soap_note(transcript)
                st.session_state.soap_note = soap_note
                
                # Add to history
                st.session_state.analysis_history.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'type': 'SOAP Note',
                    'patient': 'Patient Record'
                })
            
            st.success("✅ SOAP Note Generated!")
        else:
            st.warning("⚠️ Please enter a transcript first.")
    
    # Display SOAP note
    if 'soap_note' in st.session_state:
        st.markdown("---")
        soap = st.session_state.soap_note
        
        # Tabs for different sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📄 Full Note", "S - Subjective", "O - Objective", "A - Assessment", "P - Plan"])
        
        with tab1:
            st.markdown("### Complete SOAP Note")
            
            st.markdown(f"""
            <div class="result-box">
            <h4>S - SUBJECTIVE</h4>
            <p><b>Chief Complaint:</b> {soap['subjective']['chief_complaint']}</p>
            <p><b>History:</b> {soap['subjective']['history_of_present_illness']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="result-box">
            <h4>O - OBJECTIVE</h4>
            <p><b>Physical Exam:</b> {soap['objective']['physical_exam']}</p>
            <p><b>Observations:</b> {soap['objective']['observations']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="result-box">
            <h4>A - ASSESSMENT</h4>
            <p><b>Diagnosis:</b> {soap['assessment']['diagnosis']}</p>
            <p><b>Severity:</b> {soap['assessment']['severity']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="result-box">
            <h4>P - PLAN</h4>
            <p><b>Treatment:</b> {soap['plan']['treatment']}</p>
            <p><b>Follow-up:</b> {soap['plan']['follow_up']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("### Subjective")
            st.markdown(f"**Chief Complaint:** {soap['subjective']['chief_complaint']}")
            st.markdown(f"**History of Present Illness:** {soap['subjective']['history_of_present_illness']}")
        
        with tab3:
            st.markdown("### Objective")
            st.markdown(f"**Physical Exam:** {soap['objective']['physical_exam']}")
            st.markdown(f"**Observations:** {soap['objective']['observations']}")
        
        with tab4:
            st.markdown("### Assessment")
            st.markdown(f"**Diagnosis:** {soap['assessment']['diagnosis']}")
            st.markdown(f"**Severity:** {soap['assessment']['severity']}")
        
        with tab5:
            st.markdown("### Plan")
            st.markdown(f"**Treatment:** {soap['plan']['treatment']}")
            st.markdown(f"**Follow-up:** {soap['plan']['follow_up']}")
        
        # Export options
        st.markdown("---")
        st.subheader("💾 Export SOAP Note")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            json_output = json.dumps(soap, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_output,
                file_name=f"soap_note_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            text_output = f"""SOAP NOTE
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUBJECTIVE:
Chief Complaint: {soap['subjective']['chief_complaint']}
History: {soap['subjective']['history_of_present_illness']}

OBJECTIVE:
Physical Exam: {soap['objective']['physical_exam']}
Observations: {soap['objective']['observations']}

ASSESSMENT:
Diagnosis: {soap['assessment']['diagnosis']}
Severity: {soap['assessment']['severity']}

PLAN:
Treatment: {soap['plan']['treatment']}
Follow-up: {soap['plan']['follow_up']}
"""
            st.download_button(
                label="📄 Download Text",
                data=text_output,
                file_name=f"soap_note_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col3:
            st.button("🖨️ Print Note", use_container_width=True, help="Use browser print function (Ctrl+P)")

elif page == "📊 Analytics Dashboard":
    st.header("📊 Analytics Dashboard")
    st.markdown("Comprehensive overview of analysis patterns and insights.")
    
    if not st.session_state.analysis_history:
        st.info("👋 No analysis data yet. Start analyzing transcripts to see insights here!")
    else:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Analyses", len(st.session_state.analysis_history))
        
        with col2:
            summary_count = sum(1 for a in st.session_state.analysis_history if a['type'] == 'Medical Summary')
            st.metric("Medical Summaries", summary_count)
        
        with col3:
            soap_count = sum(1 for a in st.session_state.analysis_history if a['type'] == 'SOAP Note')
            st.metric("SOAP Notes", soap_count)
        
        with col4:
            st.metric("Last Analysis", st.session_state.analysis_history[-1]['timestamp'])
        
        st.markdown("---")
        
        # Analysis history table
        st.subheader("📋 Analysis History")
        
        history_df = pd.DataFrame(st.session_state.analysis_history)
        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Analysis type distribution
        st.markdown("---")
        st.subheader("📊 Analysis Type Distribution")
        
        type_counts = history_df['type'].value_counts()
        
        fig = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title="Analysis Types",
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Timeline
        if len(st.session_state.analysis_history) > 1:
            st.markdown("---")
            st.subheader("📈 Analysis Timeline")
            
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            history_df['date'] = history_df['timestamp'].dt.date
            
            timeline_data = history_df.groupby(['date', 'type']).size().reset_index(name='count')
            
            fig = px.line(
                timeline_data,
                x='date',
                y='count',
                color='type',
                title='Analysis Activity Over Time',
                markers=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Export history
        st.markdown("---")
        csv_data = history_df.to_csv(index=False)
        st.download_button(
            label="📥 Export History (CSV)",
            data=csv_data,
            file_name=f"analysis_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem 0;">
        <p><strong>Medical Transcript Analysis System v1.0</strong></p>
        <p>Powered by spaCy, Transformers & Streamlit</p>
        <p style="font-size: 0.8rem;">⚠️ For demonstration purposes only. Not approved for clinical use.</p>
    </div>
""", unsafe_allow_html=True)