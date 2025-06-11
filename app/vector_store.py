import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import os

class EmailVectorStore:
    def __init__(self, persist_directory: str = "email_db"):
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path="data/chroma"  # Use the existing chroma folder
        )
        
        # Create or get the collection for emails
        self.collection = self.client.get_or_create_collection(
            name="emails",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize the sentence transformer model
        self.embed_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    def prepare_email_text(self, email: Dict) -> str:
        """Prepare email text for embedding."""
        return f"""Subject: {email.get('subject', '')}
From: {email.get('sender', '')}
Date: {email.get('timestamp', '')}

Content:
{email.get('body', '')}

Summary:
{email.get('summary', '')}"""
    
    def add_email(self, email: Dict) -> None:
        """Add a single email to the vector store."""
        try:
            email_text = self.prepare_email_text(email)
            
            # Add the document to the collection
            self.collection.add(
                documents=[email_text],
                metadatas=[{
                    'id': str(email.get('id')),
                    'subject': email.get('subject'),
                    'sender': email.get('sender'),
                    'timestamp': str(email.get('timestamp')),
                    'categories': email.get('categories', [])
                }],
                ids=[str(email.get('id'))]
            )
        except Exception as e:
            print(f"Error adding email to vector store: {e}")
    
    def add_emails(self, emails: List[Dict]) -> None:
        """Add multiple emails to the vector store."""
        try:
            documents = []
            metadatas = []
            ids = []
            
            for email in emails:
                documents.append(self.prepare_email_text(email))
                metadatas.append({
                    'id': str(email.get('id')),
                    'subject': email.get('subject'),
                    'sender': email.get('sender'),
                    'timestamp': str(email.get('timestamp')),
                    'categories': email.get('categories', [])
                })
                ids.append(str(email.get('id')))
            
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Successfully added {len(emails)} emails to vector store")
        except Exception as e:
            print(f"Error adding emails to vector store: {e}")
    
    def search_emails(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for similar emails using semantic search and metadata filtering."""
        try:
            # First try to find exact matches in metadata
            metadata_results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where={"$or": [
                    {"sender": {"$contains": query}},
                    {"subject": {"$contains": query}}
                ]},
                include=["metadatas", "documents", "distances"]
            )
            
            # If no metadata matches, fall back to semantic search
            if not metadata_results['ids'][0]:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    include=["metadatas", "documents", "distances"]
                )
            else:
                results = metadata_results
            
            # Format results
            search_results = []
            for idx, (metadata, document, distance) in enumerate(zip(
                results['metadatas'][0],
                results['documents'][0],
                results['distances'][0]
            )):
                search_results.append({
                    'metadata': metadata,
                    'content': document,
                    'similarity_score': 1 - distance  # Convert distance to similarity score
                })
            
            return search_results
        except Exception as e:
            print(f"Error searching emails: {e}")
            return []

    def persist(self) -> None:
        """Persist the vector store to disk."""
        self.client.persist()