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
            max_tokens=300
        )
        
        # Initialize data retrieval agent
        self.retrieval_agent = DataRetrievalAgent()
        
        # Initialize conversation memory (stores chat history)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="output"
        )
        self.conversation_turns = {}      # Track turns per patient
        
        # Build system prompt
        self.system_prompt = self._build_system_prompt()
        
        # Initialize conversational chain
        self.qa_chain = None
        self._build_qa_chain()
        
        logger.info(f"LungScopeChatbot initialized with model: {model_name}")
    
    def _build_system_prompt(self) -> str:
        """Construct concise medical consultation prompt"""
        return """You are Dr. LungScope, an AI respiratory health assistant.

    **CRITICAL RULES:**
    1. **ALWAYS start first response by acknowledging patient's medical history** (if any)
    Example: "I see you have hypertension and are a former smoker. Let's discuss your symptoms."

    2. **If patient has pre-existing respiratory condition (COPD, Asthma, Chronic Bronchitis):**
    - Do NOT request audio recording
    - Provide diagnosis/advice based on known condition + current symptoms
    - Example: "Given your history of COPD, your current symptoms align with an exacerbation..."

    3. **If NO known respiratory condition:**
    - Ask symptom questions (1-2 turns)
    - Request audio recording (turn 3+)
    - Analyze audio and provide diagnosis

    **CONSULTATION FLOW:**

    **Turn 1 - Initial Response:**
    - Acknowledge medical history FIRST: "I see from your profile that you have [condition] and [risk factors]..."
    - Then ask about current symptoms

    **Turn 2-3 - Follow-up:**
    - If known respiratory disease exists → provide advice/diagnosis without audio
    - If NO known respiratory disease → ask for audio recording

    **Turn 4+ - After Audio:**
    - Reference medical history in diagnosis
    - Format: "Your audio shows [DISEASE] with [XX]% confidence. Given your [medical history], I recommend [action]."

    **RESPONSE RULES:**
    - Keep responses 2-4 sentences maximum
    - Check conversation history - don't repeat yourself
    - Reference known conditions when relevant

    **AUDIO REQUEST FORMAT:**
    "To accurately assess your condition, please record your lung sounds using the audio recording feature below."

    **DETECTABLE CONDITIONS (from audio):**
    Bronchiectasis, Bronchiolitis, COPD, Pneumonia, URTI, Healthy lungs

    **PRE-EXISTING CONDITIONS THAT SKIP AUDIO:**
    COPD, Asthma, Chronic Bronchitis, Bronchiectasis (if already diagnosed)

    Remember: Medical history FIRST, then symptoms or audio analysis!"""

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
        Process user query with full RAG pipeline and conversation memory
        
        Args:
            user_query: User's question or concern
            patient_id: Patient identifier (anonymized)
            audio_result: Audio classification output from Audio Analysis Agent
            
        Returns:
            Structured response with answer, confidence, and follow-up
        """
        try:
            # Track conversation turns per patient
            if patient_id not in self.conversation_turns:
                self.conversation_turns[patient_id] = 0
            self.conversation_turns[patient_id] += 1
            turn_count = self.conversation_turns[patient_id]
            
            logger.info(f"Processing query for patient: {patient_id[:6]}*** (Turn {turn_count})")
            
            # Step 1: Retrieve full context
            disease_classification = audio_result.get('disease', 'Unknown') if audio_result else 'Unknown'
            
            full_context = self.retrieval_agent.retrieve_full_context(
                patient_id=patient_id,
                disease_classification=disease_classification,
                user_query=user_query,
                audio_result=audio_result
            )
            
            # Add turn count to context
            full_context['turn_count'] = turn_count
            full_context['has_audio'] = audio_result is not None
            
            # Step 2: Format context for prompt
            patient_context_str = self._format_patient_context(full_context)
            audio_analysis_str = self._format_audio_analysis(audio_result) if audio_result else "No audio analysis provided yet"
            
            # Step 3: Query LLM with context AND conversation history
            response = self._query_with_context(
                question=user_query,
                patient_context=patient_context_str,
                audio_analysis=audio_analysis_str,
                disease_knowledge=full_context.get('disease_knowledge', []),
                turn_count=turn_count,
                has_audio=audio_result is not None
            )
            
            # Step 4: Generate follow-up suggestions
            follow_up = self._generate_follow_up(disease_classification, audio_result, turn_count)
            
            # Step 5: Calculate confidence
            confidence = self._calculate_response_confidence(audio_result, full_context)
            
            # Step 6: Structure response
            structured_response = {
                "response": response,
                "confidence": float(confidence),
                "disease_classification": disease_classification,
                "follow_up": follow_up,
                "turn_count": turn_count,
                "should_request_audio": turn_count >= 3 and not audio_result,
                "timestamp": datetime.now().isoformat(),
            }
            
            # Step 7: Log interaction to audit trail
            self.retrieval_agent.log_interaction(
                patient_id=patient_id,
                query=user_query,
                response=response,
                audio_result=audio_result
            )
            
            logger.info(f"Query processed successfully for patient: {patient_id[:6]}*** (Turn {turn_count})")
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
                "turn_count": 0,
                "should_request_audio": False,
                "timestamp": datetime.now().isoformat()
            }

    
    def _query_with_context(
        self,
        question: str,
        patient_context: str,
        audio_analysis: str,
        disease_knowledge: List[Dict],
        turn_count: int,
        has_audio: bool
    ) -> str:
        """Query LLM with injected context AND conversation history"""
        
        # Get conversation history from memory
        chat_history = self.memory.load_memory_variables({})
        history_str = ""
        
        if chat_history.get("chat_history"):
            history_str = "\n\n**PREVIOUS CONVERSATION:**\n"
            for msg in chat_history["chat_history"]:
                # Check message type properly
                if hasattr(msg, 'type'):
                    role = "Patient" if msg.type == "human" else "Dr. LungScope"
                else:
                    role = "Dr. LungScope"
                
                content = msg.content if hasattr(msg, 'content') else str(msg)
                history_str += f"{role}: {content}\n"
            
            history_str += "\n**CRITICAL:** Review the above conversation. DO NOT repeat medical history or ask the same questions again!\n"
        
        # Format knowledge context (limit to top 3 sources)
        knowledge_context = ""
        if disease_knowledge:
            knowledge_context = "\n".join([
                f"Medical Source {idx + 1}: {doc.get('content', '')[:150]}..."
                for idx, doc in enumerate(disease_knowledge[:3])
            ])
        else:
            knowledge_context = "No specific disease knowledge retrieved."
        
        # Audio request logic
        audio_prompt = ""
        if turn_count >= 3 and not has_audio:
            audio_prompt = '\n**ACTION REQUIRED:** We\'ve discussed symptoms for 3+ turns. Now say: "To provide accurate diagnosis, please record your lung sounds using the audio recording button below."\n'
        
        # Build full prompt WITH HISTORY
        full_prompt = f"""{self.system_prompt}

    **CONTEXT FOR THIS CONSULTATION:**
    Turn Number: {turn_count}/10
    Audio Provided: {"Yes - Analyze and provide diagnosis" if has_audio else "No - Ask for recording if turn >= 3"}

    Medical Knowledge Available:
    {knowledge_context}

    {patient_context}

    {audio_analysis}
    {audio_prompt}
    {history_str}

    **CURRENT PATIENT QUESTION:** {question}

    **YOUR RESPONSE (2-4 sentences ONLY, acknowledge history):**
    """
        
        # Invoke LLM
        response = self.llm.invoke(full_prompt)
        response_text = response.content.strip()
        
        # CRITICAL: Save to memory with correct keys
        self.memory.save_context(
            {"input": question},
            {"output": response_text}
        )
        
        return response_text


    
    def _generate_follow_up(
        self,
        disease: str,
        audio_result: Optional[Dict],
        turn_count: int
    ) -> str:
        """Generate contextual follow-up based on conversation progress"""
        
        # If audio provided, give diagnosis-based follow-up
        if audio_result:
            confidence = audio_result.get('confidence', 0.0)
            if confidence > 0.8:
                return "Schedule follow-up with pulmonologist if symptoms worsen."
            else:
                return "Audio analysis inconclusive. Recommend in-person examination."
        
        # If no audio and turn >= 3, request audio
        if turn_count >= 3:
            return "Please record lung sounds using the audio recording feature for accurate diagnosis."
        
        # Early turns: gather more info
        if turn_count == 1:
            return "Please provide more details about your symptoms."
        elif turn_count == 2:
            return "Any other symptoms or changes you've noticed recently?"
        else:
            return "Continue describing symptoms for better assessment."

    
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
