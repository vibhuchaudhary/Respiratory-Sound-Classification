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
        Construct HIPAA-compliant system prompt for medical assistant
        
        Returns:
            System prompt string
        """
        prompt = """You are LungScope AI, a knowledgeable medical assistant specializing in respiratory health analysis.

Your role is to:
1. Provide helpful, evidence-based information about respiratory diseases
2. Explain audio analysis results in accessible language
3. Offer personalized insights based on patient medical history
4. Recommend appropriate next steps and follow-up care

CRITICAL CONSTRAINTS:
- You are an AI assistant, NOT a licensed physician
- Always emphasize that your analysis supports, but does not replace, professional medical diagnosis
- Never make definitive diagnoses - use phrases like "based on the audio analysis" or "the data suggests"
- Respect patient privacy - never share or reference raw PHI
- If audio confidence is below 85%, strongly recommend in-person evaluation
- Always recommend consulting a pulmonologist for severe symptoms

DISEASE CLASSIFICATIONS YOU WORK WITH:
- Bronchiectasis: Chronic condition with widened airways
- Bronchiolitis: Inflammation of small airways
- COPD: Chronic Obstructive Pulmonary Disease
- Pneumonia: Lung infection
- URTI: Upper Respiratory Tract Infection
- Healthy: No respiratory abnormalities detected

RESPONSE FORMAT:
- Start with a compassionate acknowledgment of the patient's concern
- Explain the audio analysis result and what it means
- Integrate patient medical history (comorbidities, medications, allergies)
- Provide actionable recommendations
- Suggest follow-up questions or next steps
- End with encouragement to seek professional care when appropriate

Use the provided context (patient history, disease knowledge, medical guidelines) to give personalized, accurate advice.
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
                "confidence": confidence,
                "disease_classification": disease_classification,
                "follow_up": follow_up,
                "timestamp": datetime.now().isoformat(),
                "patient_id": patient_id[:6] + "***"  # Partial anonymization for logging
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
            return {
                "response": "I apologize, but I encountered an error processing your request. Please try again or contact support.",
                "confidence": 0.0,
                "follow_up": "Please retry your query or seek in-person medical assistance.",
                "error": str(e)
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
        response = self.llm.predict(full_prompt)
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
    chatbot = LungScopeChatbot(temperature=0.3, model_name="gpt-4")
    
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
