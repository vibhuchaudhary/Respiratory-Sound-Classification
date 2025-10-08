"""
data_ingest.py - Medical Knowledge Base Ingestion for A.I.R.A
Loads PDF/text medical documents into ChromaDB with disease-specific metadata
"""

import os
import logging
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


class MedicalDataIngestor:
    """
    Ingest medical documents into ChromaDB with disease-specific metadata tagging
    """
    
    DISEASE_CATEGORIES = [
        "Bronchiectasis",
        "Bronchiolitis",
        "COPD",
        "Healthy",
        "Pneumonia",
        "URTI"
    ]
    
    CONTENT_TYPES = [
        "symptoms",
        "treatment",
        "diagnosis",
        "complications",
        "prevention",
        "general"
    ]
    
    def __init__(
        self,
        data_directory: str = "./data",
        vector_db_path: str = "./vector_db/chroma_db",
        collection_name: str = "medical_knowledge"
    ):
        """
        Initialize medical data ingestor
        
        Args:
            data_directory: Path to medical documents (PDFs, txt files)
            vector_db_path: Path to persist ChromaDB
            collection_name: ChromaDB collection name
        """
        self.data_directory = data_directory
        self.vector_db_path = vector_db_path
        self.collection_name = collection_name
        
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            model="text-embedding-3-small"
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200, 
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        logger.info(f"MedicalDataIngestor initialized")
        logger.info(f"Data directory: {data_directory}")
        logger.info(f"Vector DB path: {vector_db_path}")
    
    def load_documents(self, file_pattern: str = "*.pdf") -> List[Document]:
        """
        Load medical documents from data directory
        
        Args:
            file_pattern: File pattern to match (*.pdf, *.txt, etc.)
            
        Returns:
            List of LangChain Document objects
        """
        try:
            documents = []
            
            if file_pattern == "*.pdf":
                loader = DirectoryLoader(
                    self.data_directory,
                    glob=file_pattern,
                    loader_cls=PyPDFLoader,
                    show_progress=True
                )
                documents.extend(loader.load())
                logger.info(f"Loaded {len(documents)} PDF documents")
            
            elif file_pattern == "*.txt":
                loader = DirectoryLoader(
                    self.data_directory,
                    glob=file_pattern,
                    loader_cls=TextLoader,
                    show_progress=True
                )
                documents.extend(loader.load())
                logger.info(f"Loaded {len(documents)} text documents")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return []
    
    def extract_disease_from_content(self, text: str) -> str:
        """
        Extract disease category from document content
        
        Args:
            text: Document text
            
        Returns:
            Disease category or 'general'
        """
        text_lower = text.lower()
        
        for disease in self.DISEASE_CATEGORIES:
            if disease.lower() in text_lower:
                return disease
        
        return "general"
    
    def extract_content_type(self, text: str) -> str:
        """
        Determine content type from document text
        
        Args:
            text: Document text
            
        Returns:
            Content type category
        """
        text_lower = text.lower()
        
        # Check for content type keywords
        if any(kw in text_lower for kw in ["symptom", "sign", "manifestation"]):
            return "symptoms"
        elif any(kw in text_lower for kw in ["treatment", "therapy", "medication", "drug"]):
            return "treatment"
        elif any(kw in text_lower for kw in ["diagnosis", "diagnostic", "test", "examination"]):
            return "diagnosis"
        elif any(kw in text_lower for kw in ["complication", "risk", "prognosis"]):
            return "complications"
        elif any(kw in text_lower for kw in ["prevention", "preventive", "avoid"]):
            return "prevention"
        else:
            return "general"
    
    def add_metadata(self, documents: List[Document]) -> List[Document]:
        """
        Add disease-specific metadata to document chunks
        
        Args:
            documents: List of documents
            
        Returns:
            Documents with enriched metadata
        """
        enriched_docs = []
        
        for doc in documents:
            disease = self.extract_disease_from_content(doc.page_content)
            content_type = self.extract_content_type(doc.page_content)
            doc.metadata.update({
                "disease": disease,
                "content_type": content_type,
                "ingestion_date": str(Path(doc.metadata.get("source", "")).stat().st_mtime) if "source" in doc.metadata else "unknown",
                "chunk_length": len(doc.page_content)
            })
            
            enriched_docs.append(doc)
        
        logger.info(f"Added metadata to {len(enriched_docs)} documents")
        return enriched_docs
    
    def ingest_to_vectordb(self, documents: List[Document]) -> Chroma:
        """
        Ingest documents into ChromaDB with embeddings
        
        Args:
            documents: List of documents with metadata
            
        Returns:
            ChromaDB vectorstore
        """
        try:
            logger.info("Splitting documents into chunks...")
            text_chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(text_chunks)} text chunks")
            
            enriched_chunks = self.add_metadata(text_chunks)
            
            logger.info("Creating embeddings and ingesting into ChromaDB...")
            vectorstore = Chroma.from_documents(
                documents=enriched_chunks,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=self.vector_db_path
            )
            
            logger.info(f"Successfully ingested {len(enriched_chunks)} chunks into ChromaDB")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Error ingesting documents: {e}")
            raise
    
    def run_ingestion(self, file_patterns: List[str] = ["*.pdf", "*.txt"]):
        """
        Run complete ingestion pipeline
        
        Args:
            file_patterns: List of file patterns to process
        """
        logger.info("Starting medical knowledge ingestion pipeline...")
        
        all_documents = []
        
        for pattern in file_patterns:
            docs = self.load_documents(file_pattern=pattern)
            all_documents.extend(docs)
        
        if not all_documents:
            logger.warning("No documents found to ingest")
            return
        
        logger.info(f"Total documents loaded: {len(all_documents)}")
        
        vectorstore = self.ingest_to_vectordb(all_documents)
        
        logger.info("Ingestion pipeline completed successfully!")
        
        self.display_stats(vectorstore)
    
    def display_stats(self, vectorstore: Chroma):
        """Display ingestion statistics"""
        try:
            collection = vectorstore._collection
            count = collection.count()
            
            logger.info("\n" + "="*50)
            logger.info("INGESTION STATISTICS")
            logger.info("="*50)
            logger.info(f"Total chunks in database: {count}")
            logger.info(f"Collection name: {self.collection_name}")
            logger.info(f"Embedding model: text-embedding-3-small")
            logger.info(f"Vector DB location: {self.vector_db_path}")
            logger.info("="*50 + "\n")
            
        except Exception as e:
            logger.error(f"Error displaying stats: {e}")


def main():
    """Main function to run ingestion"""
    ingestor = MedicalDataIngestor(
        data_directory="./data/medical_knowledge",
        vector_db_path="./vector_db/chroma_db",
        collection_name="medical_knowledge"
    )
    
    ingestor.run_ingestion(file_patterns=["*.pdf", "*.txt"])


if __name__ == "__main__":
    main()
