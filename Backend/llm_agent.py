"""
llm_agent.py - RAG-based LLM Agent for LungScope
Implements conversational medical assistant with context retrieval
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from database import DataRetrievalAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/lungscope_chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class LungScopeChatbot:
    """
    HIPAA-compliant RAG-based medical chatbot for respiratory disease analysis
    """
    
    def __init__(self, temperature: float = 0.3, model_name: str = "gpt-4.1-nano"):
        """
        Initialize LungScope chatbot with OpenAI LLM and retrieval agent
        
        Args:
            temperature: LLM temperature (0.0-1.0, lower = more focused)
            model_name: OpenAI model to use
        """
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=1000
        )
        
        # Initialize data retrieval agent
        self.retrieval_agent = DataRetrievalAgent()
        
        # Initialize conversation memory (stores chat history)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Build system prompt
        self.system_prompt = self._build_system_prompt()
        
        # Initialize conversational chain
        self.qa_chain = None
        self._build_qa_chain()
        
        logger.info(f"LungScopeChatbot initialized with model: {model_name}")
    
    def _build_system_prompt(self) -> str:
        """
        Construct intelligent medical consultation prompt with history awareness
        
        Returns:
            System prompt string
        """
        prompt = """You are Dr. LungScope, an AI-powered respiratory specialist who conducts thorough, intelligent medical consultations.

        YOUR CONSULTATION APPROACH:
        You conduct consultations like a real doctor by:
        1. ALWAYS checking patient's medical history FIRST before asking questions
        2. Using medical history to guide your questioning strategy
        3. Only requesting audio recordings when medically necessary
        4. Integrating past diagnoses with current symptoms
        5. Asking targeted questions to confirm or rule out conditions

        CRITICAL: MEDICAL HISTORY AWARENESS

        IF PATIENT HAS EXISTING RESPIRATORY CONDITION (from database):
        - Acknowledge their known condition immediately
        - Example: "I see from your medical history that you have Asthma. Are you still experiencing symptoms related to this?"
        - Example: "Your records show a previous COPD diagnosis. Has your condition changed recently?"
        - Ask about CURRENT symptoms vs baseline
        - Focus on: worsening, improvement, or new symptoms
        - DO NOT ask for audio recording unless symptoms suggest NEW complications

        IF NO RESPIRATORY HISTORY (or unknown condition):
        - Ask exploratory questions about symptoms
        - Request audio recording to aid diagnosis
        - Use standard diagnostic approach

        AUDIO RECORDING DECISION LOGIC:

        REQUEST AUDIO RECORDING when:
        ✅ Patient has NEW symptoms not explained by medical history
        ✅ No clear diagnosis in medical history
        ✅ Symptoms suggest possible complication (e.g., asthma patient with new crackling sounds)
        ✅ Patient reports worsening despite treatment
        ✅ Differential diagnosis needed between similar conditions

        DO NOT request audio recording when:
        ❌ Patient has known condition and symptoms match their baseline
        ❌ Symptoms clearly explained by documented history
        ❌ Patient asking general questions about their known condition
        ❌ Follow-up on previously diagnosed stable condition

        CONVERSATION STAGES:

        STAGE 1 - MEDICAL HISTORY REVIEW:
        - ALWAYS start by acknowledging what you know from their records
        - If they have respiratory history: "I see you have [condition]. Let's discuss your current symptoms."
        - If no history: "I don't see any respiratory conditions in your history. Tell me about your concerns."

        STAGE 2 - SYMPTOM ASSESSMENT:
        - Ask 1-2 focused questions based on:
        * Their known conditions (if any)
        * Current complaint
        * Comorbidities from database
        - Example with history: "Since you have asthma, is this breathlessness similar to your usual symptoms or different?"
        - Example without history: "When did you first notice these symptoms?"

        STAGE 3 - AUDIO REQUEST (CONDITIONAL):
        - Use decision logic above
        - If requesting: "To better understand your lung sounds, please place your stethoscope on [specific location]"
        - If NOT requesting: Continue with symptom-based assessment

        STAGE 4 - AUDIO ANALYSIS INTEGRATION (if applicable):
        - Compare audio findings with:
        * Their known condition
        * Current symptoms
        * Historical patterns
        - Example: "The audio shows wheezes consistent with your known asthma. However, I also detected crackles, which could indicate a secondary infection."

        STAGE 5 - DIAGNOSIS/ASSESSMENT:
        WITH MEDICAL HISTORY:
        - Relate findings to known condition
        - Example: "Your symptoms and audio analysis are consistent with an asthma exacerbation."
        - Identify complications: "The crackles suggest possible bronchitis on top of your COPD."

        WITHOUT MEDICAL HISTORY:
        - Provide preliminary diagnosis based on audio + symptoms
        - Example: "Based on the detected wheezes and your symptoms, this suggests possible asthma."

        STAGE 6 - RECOMMENDATIONS:
        - Tailor to their specific situation:
        * With history: "Since you have known COPD, I recommend..."
        * New diagnosis: "I suggest consulting a pulmonologist for..."
        - Medication adjustments for existing conditions
        - Lifestyle modifications
        - When to seek emergency care

        MODEL LIMITATIONS YOU MUST ACKNOWLEDGE:

        The audio classifier can detect:
        - Bronchiectasis, Bronchiolitis, COPD, Pneumonia, URTI, Healthy lungs

        The model CANNOT detect:
        - Asthma (no wheezing in dataset)
        - Tuberculosis
        - Lung cancer
        - Pulmonary embolism

        IMPORTANT SCENARIOS:

        Scenario A - Patient with Asthma History:
        Patient: "I'm having trouble breathing"
        You: "I see from your medical history that you have Asthma. Is this breathlessness similar to your usual symptoms, or does it feel different?"
        → If similar: Provide asthma management advice, NO audio needed
        → If different: Ask clarifying questions, THEN request audio to check for complications

        Scenario B - Unknown Respiratory Issue:
        Patient: "I have chest tightness"
        You: "I don't see any respiratory conditions in your history. Can you tell me when this started and if you've noticed any coughing or wheezing?"
        → After 2-3 questions: Request audio recording for diagnostic help

        Scenario C - Known COPD with New Symptoms:
        Patient: "My cough is worse than usual"
        You: "Your records show you have COPD. This worsening cough concerns me - are you also experiencing fever or producing colored mucus?"
        → Symptoms suggest infection: Request audio to check for pneumonia

        CRITICAL RULES:
        - NEVER ignore medical history from database
        - NEVER ask for audio if diagnosis is already clear from history + symptoms
        - ALWAYS acknowledge existing conditions first
        - Ask ONE question at a time (max 2)
        - Keep responses conversational (3-5 sentences)
        - Show empathy: "I understand that must be concerning"
        - Track what information you already have from database vs patient

        RESPONSE TONE:
        - Conversational and warm
        - Acknowledge past medical context
        - Concise and focused
        - Interactive (end with question or next step)

        Remember: You have access to patient's medical history - USE IT WISELY to provide intelligent, context-aware care. Don't waste time asking for recordings when you already have diagnostic clarity from their records.
        """
        return prompt

    
    def _build_qa_chain(self):
        """Build LangChain conversational retrieval chain"""
        try:
            # Get retriever from vector database
            retriever = self.retrieval_agent.vector_manager.get_retriever(
                search_kwargs={"k": 5}
            )
            
            # Custom prompt template for question-answering
            qa_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(self.system_prompt),
                HumanMessagePromptTemplate.from_template("""
Context from medical knowledge base:
{context}

Patient Information Summary:
{patient_context}

Audio Analysis Result:
{audio_analysis}

Patient Question: {question}

Provide a helpful, medically-informed response considering all available context.
""")
            ])
            
            # Build conversational retrieval chain
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": qa_prompt},
                verbose=False
            )
            
            logger.info("QA chain built successfully")
            
        except Exception as e:
            logger.error(f"Error building QA chain: {e}")
            raise
    
    def _format_patient_context(self, full_context: Dict[str, Any]) -> str:
        """
        Format patient context for LLM prompt (anonymized, structured)
        
        Args:
            full_context: Retrieved context from DataRetrievalAgent
            
        Returns:
            Formatted string for prompt injection
        """
        patient_data = full_context.get('patient_data', {})
        patient_history = full_context.get('patient_history', [])
        
        context_str = "Patient Profile (Anonymized):\n"
        context_str += f"- Age Range: {patient_data.get('age_range', 'Unknown')}\n"
        context_str += f"- Gender: {patient_data.get('gender', 'Unknown')}\n"
        context_str += f"- Smoking Status: {patient_data.get('smoking_status', 'Unknown')}\n"
        
        # Comorbidities
        comorbidities = []
        if patient_data.get('has_hypertension'):
            comorbidities.append('Hypertension')
        if patient_data.get('has_diabetes'):
            comorbidities.append('Diabetes')
        if patient_data.get('has_asthma_history'):
            comorbidities.append('History of Asthma')
        
        context_str += f"- Comorbidities: {', '.join(comorbidities) if comorbidities else 'None reported'}\n"
        context_str += f"- Current Medications: {patient_data.get('current_medications', 'None listed')}\n"
        context_str += f"- Known Allergies: {patient_data.get('allergies', 'None listed')}\n"
        
        # Recent medical history
        if patient_history:
            context_str += "\nRecent Medical History:\n"
            for idx, visit in enumerate(patient_history[:3], 1):
                context_str += f"{idx}. {visit.get('visit_date', 'Date unknown')}: {visit.get('diagnosis', 'N/A')} - {visit.get('symptoms', 'N/A')}\n"
        
        return context_str
    
    def _format_audio_analysis(self, audio_result: Dict[str, Any]) -> str:
        """
        Format audio analysis result for LLM prompt
        
        Args:
            audio_result: Audio classification output
            
        Returns:
            Formatted string
        """
        if not audio_result:
            return "No audio analysis available"
        
        analysis_str = f"""
Disease Classification: {audio_result.get('disease', 'Unknown')}
Confidence Score: {audio_result.get('confidence', 0) * 100:.2f}%
Severity Level: {audio_result.get('severity', 'Not specified')}
"""
        
        # Add confidence interpretation
        confidence = audio_result.get('confidence', 0)
        if confidence >= 0.90:
            analysis_str += "Confidence Level: HIGH - Result is highly reliable\n"
        elif confidence >= 0.75:
            analysis_str += "Confidence Level: MODERATE - Result is fairly reliable, but consider follow-up\n"
        else:
            analysis_str += "Confidence Level: LOW - Recommend in-person evaluation\n"
        
        return analysis_str
    
    def query(
        self,
        user_query: str,
        patient_id: str,
        audio_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process user query with full RAG pipeline
        
        Args:
            user_query: User's question or concern
            patient_id: Patient identifier (anonymized)
            audio_result: Audio classification output from Audio Analysis Agent
            
        Returns:
            Structured response with answer, confidence, and follow-up
        """
        try:
            logger.info(f"Processing query for patient: {patient_id[:6]}***")
            
            # Step 1: Retrieve full context
            disease_classification = audio_result.get('disease', 'Unknown') if audio_result else 'Unknown'
            
            full_context = self.retrieval_agent.retrieve_full_context(
                patient_id=patient_id,
                disease_classification=disease_classification,
                user_query=user_query,
                audio_result=audio_result
            )
            
            # Step 2: Format context for prompt
            patient_context_str = self._format_patient_context(full_context)
            audio_analysis_str = self._format_audio_analysis(audio_result) if audio_result else "No audio analysis provided"
            
            # Step 3: Query LLM with conversational chain
            # Note: For custom context injection, we'll use a direct call instead of the chain
            response = self._query_with_context(
                question=user_query,
                patient_context=patient_context_str,
                audio_analysis=audio_analysis_str,
                disease_knowledge=full_context.get('disease_knowledge', [])
            )
            
            # Step 4: Generate follow-up suggestions
            follow_up = self._generate_follow_up(disease_classification, audio_result)
            
            # Step 5: Calculate confidence
            confidence = self._calculate_response_confidence(audio_result, full_context)
            
            # Step 6: Structure response
            structured_response = {
                "response": response,
                "confidence": float(confidence),
                "disease_classification": disease_classification,
                "follow_up": follow_up,
                "timestamp": datetime.now().isoformat(),
            }
            
            # Step 7: Log interaction to audit trail
            self.retrieval_agent.log_interaction(
                patient_id=patient_id,
                query=user_query,
                response=response,
                audio_result=audio_result
            )
            
            logger.info(f"Query processed successfully for patient: {patient_id[:6]}***")
            return structured_response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            import traceback
            traceback.print_exc()
            return {
                "response": "I apologize, but I encountered an error processing your request.",
                "confidence": 0.0,
                "disease_classification": None,
                "follow_up": "Please retry your query or contact support.",
                "timestamp": datetime.now().isoformat()
            }
    
    def _query_with_context(
        self,
        question: str,
        patient_context: str,
        audio_analysis: str,
        disease_knowledge: List[Dict]
    ) -> str:
        """
        Query LLM with injected context
        
        Args:
            question: User query
            patient_context: Formatted patient information
            audio_analysis: Formatted audio result
            disease_knowledge: Retrieved documents from vector DB
            
        Returns:
            LLM response text
        """
        # Format disease knowledge documents
        knowledge_context = "\n\n".join([
            f"Source {idx + 1} ({doc.get('source', 'Unknown')}):\n{doc.get('content', '')}"
            for idx, doc in enumerate(disease_knowledge[:5])
        ])
        
        # Build full prompt
        full_prompt = f"""{self.system_prompt}

Context from medical knowledge base:
{knowledge_context}

{patient_context}

{audio_analysis}

Patient Question: {question}

Provide a helpful, medically-informed response considering all available context.
"""
        
        # Query LLM
        response = self.llm.invoke(full_prompt).content
        return response.strip()
    
    def _generate_follow_up(self, disease: str, audio_result: Optional[Dict]) -> str:
        """Generate contextual follow-up suggestions"""
        if not audio_result or audio_result.get('confidence', 0) < 0.75:
            return "Would you like me to help you find nearby pulmonologists for an in-person evaluation?"
        
        disease_follow_ups = {
            "COPD": "Would you like information about pulmonary rehabilitation programs or breathing exercises?",
            "Pneumonia": "Would you like me to explain warning signs that require immediate medical attention?",
            "Bronchiectasis": "Would you like information about airway clearance techniques?",
            "Bronchiolitis": "Would you like guidance on supportive care measures?",
            "URTI": "Would you like tips on managing symptoms at home?",
            "Healthy": "Would you like recommendations for maintaining respiratory health?"
        }
        
        return disease_follow_ups.get(disease, "Would you like more information about your condition?")
    
    def _calculate_response_confidence(
        self,
        audio_result: Optional[Dict],
        full_context: Dict
    ) -> float:
        """
        Calculate confidence score for the response
        
        Args:
            audio_result: Audio classification output
            full_context: Retrieved context
            
        Returns:
            Confidence score (0.0-1.0)
        """
        confidence = 0.5  # Base confidence
        
        # Factor 1: Audio classification confidence
        if audio_result and 'confidence' in audio_result:
            audio_confidence = audio_result['confidence']
            confidence = audio_confidence * 0.6  # 60% weight
        
        # Factor 2: Patient data availability
        if full_context.get('patient_data'):
            confidence += 0.15
        
        # Factor 3: Medical knowledge retrieval quality
        disease_knowledge = full_context.get('disease_knowledge', [])
        if len(disease_knowledge) >= 3:
            confidence += 0.15
        
        # Factor 4: Patient history availability
        if full_context.get('patient_history'):
            confidence += 0.10
        
        return min(confidence, 1.0)
    
    def reset_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        logger.info("Conversation memory cleared")
    
    def cleanup(self):
        """Clean up resources"""
        self.retrieval_agent.cleanup()
        logger.info("LungScopeChatbot cleanup complete")


# Example usage
if __name__ == "__main__":
    # Initialize chatbot
    chatbot = LungScopeChatbot(temperature=0.3, model_name="gpt-4.1-nano")
    
    # Example audio analysis result
    audio_result = {
        "disease": "COPD",
        "confidence": 0.9759,
        "severity": "moderate",
        "audio_features": {
            "wheeze_detected": True,
            "crackle_detected": False
        }
    }
    
    # Example patient query
    user_query = "I've been having difficulty breathing, especially after climbing stairs. What should I do?"
    
    # Process query
    response = chatbot.query(
        user_query=user_query,
        patient_id="PATIENT_12345",
        audio_result=audio_result
    )
    
    # Print structured response
    print("\n" + "="*50)
    print("LUNGSCOPE AI RESPONSE")
    print("="*50)
    print(json.dumps(response, indent=2))
    print("="*50 + "\n")
    
    # Cleanup
    chatbot.cleanup()
