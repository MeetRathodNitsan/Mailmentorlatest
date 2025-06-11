import base64
from app.models import Email
from sqlalchemy.orm import Session
from app.config import engine
import html
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime,timezone
# from ollama_client import OllamaClient
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.utils import parsedate_to_datetime
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from sympy import content
import openai
import re
from app.config import OPENAI_API_KEY
from bs4 import BeautifulSoup,Comment
from wasabi import msg
from app.config import OPENAI_API_KEY, OPENAI_API_ENDPOINT, SessionLocal
from app.models import Email
from app.ai_services import generate_ai_response, analyze_email_content, generate_ai_response_to_email
from app.utils import extract_email_content, build_query
from app.vector_store import EmailVectorStore
from app.categorization import categorize_email
import requests
from llama_index.core import Document, VectorStoreIndex
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from transformers import pipeline
import time
from transformers import AutoTokenizer, AutoModel
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


STOP_WORDS = set(stopwords.words("english"))
openai.api_key = OPENAI_API_KEY

DATABASE_URL = "postgresql://postgres:admin@localhost:5432/mailmentor"


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Email(Base):

    __tablename__ = "emails"

    id = Column(String, primary_key=True)
    sender = Column(Text)
    recipient = Column(Text)
    subject = Column(Text)
    body = Column(Text)
    timestamp = Column(DateTime)
    category = Column(Text)
    summary = Column(Text)
    ai_response = Column(Text)
    status = Column(String, default='pending')

class EmailProcessor:
    def __init__(self, credentials: Credentials):
        self.credentials = credentials
        self.service = build('gmail', 'v1', credentials=credentials)
        self.vectorizer = TfidfVectorizer()
        self.email_cache = {}
        self.last_cache_update = None
        self.vector_store = EmailVectorStore()


    def _preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r"\d+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = text.split()
        tokens = [word for word in tokens if word not in STOP_WORDS]
        return " ".join(tokens)

        
       
    def initialize_vector_store(self):
        """Initialize the vector store with existing emails."""
        try:
            emails = self.fetch_emails(limit=100)  # Adjust limit as needed
            self.vector_store.add_emails(emails)
            self.vector_store.persist()
        except Exception as e:
            print(f"Error initializing vector store: {e}")

    def search_emails_nlp(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            results = self.vector_store.search_emails(query, n_results=top_k)
        # Convert vector store results to the expected format
            formatted_results = []
            for result in results:
                email_data = result['metadata']
                email_data['score'] = result['similarity_score']
                formatted_results.append(email_data)
            return formatted_results
        except Exception as e:
            print(f"Smart search failed: {e}")
            return []

    def fetch_emails(self, limit: int = 15) -> List[Dict[str, Any]]:
        try:
            # Remove the time restriction and focus on filtering out non-personal emails
            query = "from:(*@*.* -@newsletter.* -@marketing.* "\
                    "-@promo.* -@offer.* -@deals.* -noreply* "\
                    "-no-reply* -notification* -automated* -donotreply* "\
                    "-automated* -update* -info@* -support@* -newsletter@*)"
            
            # Check local cache
            cache_key = f"emails_{limit}"
            current_time = datetime.now(timezone.utc)
            if cache_key in self.email_cache:
                cached_data = self.email_cache[cache_key]
                if (current_time - cached_data['timestamp']).seconds < 300:  # 5 min cache
                    return cached_data['emails']
            
            results = self.service.users().messages().list(
                userId='me',
                maxResults=limit * 2,  # Fetch extra to account for filtering
                q=query
            ).execute()
            
            messages = results.get('messages', [])
            emails = []
            
            for message in messages:
                email_data = self._process_single_email(message['id'])
                if email_data:
                    # Simplified person email check
                    sender_email = email_data.get('sender', '').split('<')[-1].strip('>')
                    if self._is_person_email(email_data.get('sender', ''), sender_email):
                        emails.append(email_data)
                        
                        if len(emails) >= limit:
                            break
            
            # Update local cache
            self.email_cache[cache_key] = {
                'emails': emails,
                'timestamp': current_time
            }
            
            return emails
        except Exception as e:
            print(f"Error fetching emails: {e}")
            return []
    
    def _is_person_email(self, sender_name: str, sender_email: str) -> bool:
        # Exclude common system patterns first
        system_patterns = ['noreply', 'no-reply', 'notification',
                          'donotreply', 'newsletter', 'marketing',
                          'automated', 'system']
        if any(pattern in sender_email.lower() for pattern in system_patterns):
            return False
        
        # Check for known newsletter/marketing domains
        newsletter_domains = ['newsletter.', 'marketing.', 'promo.',
                             'offer.', 'deals.', 'campaign.']
        if any(domain in sender_email.lower() for domain in newsletter_domains):
            return False
        
        # Accept all other emails that aren't filtered out above
        return True
        try:
            # Fetch messages from Gmail API
            results = self.service.users().messages().list(
                userId='me',
                maxResults=limit
            ).execute()
            
            messages = results.get('messages', [])
            emails = []
            
            for message in messages:
                email_data = self._process_single_email(message['id'])
                if email_data:
                    emails.append(email_data)
            
            return emails
        except Exception as e:
            print(f"Error fetching emails: {e}")
            return []

    def _process_single_email(self, msg_id: str) -> Optional[Dict[str, Any]]:
        try:
            email_data = self.service.users().messages().get(
                userId='me',
                id=msg_id,
                format='full'
            ).execute()
            
            headers = {h['name']: h['value'] for h in email_data['payload']['headers']}
            subject = headers.get('Subject', '(No subject)')
            sender = headers.get('From', '(Unknown sender)')
            timestamp = headers.get('Date', datetime.now().strftime('%a, %d %b %Y %H:%M:%S +0000'))
            labels = email_data.get('labelIds', [])
            priority= self._determine_priority(subject, email_data.get('snippet', ''), labels, sender)
            catagory = self._determine_category(subject, email_data.get('snippet', ''), labels, priority)
            
            # Extract content using existing method
            content_data = self._extract_email_content(email_data)
            body = content_data.get('body', '')
            ai_summary = content_data.get('ai_summary', '')
            ai_response = content_data.get('ai_response', '')
            
            # Only generate AI content for the email if it's displayed
            return {
                 'id': msg_id,
                'subject': subject,
                'sender': sender,
                'timestamp': timestamp,
                'body': body,
                'ai_summary': ai_summary,
                'ai_response': ai_response,
                'priority': priority,
                'catagory': catagory,
            }
        except Exception as e:
            print(f"Error in _process_single_email: {e}")
            return None
            
    def get_email_stats(self):
        try:
            print("Fetching email statistics...")
            
            # Get unread emails count
            unread_results = self.service.users().messages().list(
                userId='me',
                labelIds=['UNREAD'],
                maxResults=500
            ).execute()
            unread_count = len(unread_results.get('messages', []))
            print(f"Unread count: {unread_count}")
            
            # Get action items count by searching for action-related keywords
            action_query = "subject:(action OR todo OR task OR urgent OR required OR asap)"
            action_results = self.service.users().messages().list(
                userId='me',
                q=action_query,
                maxResults=500
            ).execute()
            action_items_count = len(action_results.get('messages', []))
            print(f"Action items count: {action_items_count}")
            
            # Get active threads
            threads_results = self.service.users().threads().list(
                userId='me',
                labelIds=['INBOX']
            ).execute()
            threads_count = len(threads_results.get('threads', []))
            print(f"Threads count: {threads_count}")
            
            # Get pending responses (unread in inbox)
            pending_results = self.service.users().messages().list(
                userId='me',
                q='in:inbox is:unread',
                maxResults=500
            ).execute()
            pending_count = len(pending_results.get('messages', []))
            print(f"Pending count: {pending_count}")
            
            # Calculate email categories distribution
            categories_count = {
                'General': 0,
                'Urgent': 0,
                'Action Item': 0,
                'Follow-up': 0
            }
            
            # Get recent emails to calculate categories
            recent_emails = self.fetch_emails(limit=100)  # Fetch more emails for better stats
            for email in recent_emails:
                # Fix: Use 'catagory' instead of 'category' to match the key from _process_single_email
                category = email.get('catagory', 'General')
                if category not in categories_count:
                    categories_count[category] = 0
                categories_count[category] += 1
            
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            return {
                'unread': unread_count,
                'action_items': action_items_count,
                'threads': threads_count,
                'pending': pending_count,
                'categories': categories_count,
                'last_updated': current_time
            }
        except Exception as e:
            print(f"Error getting email stats: {e}")
            return {
                'unread': 0,
                'action_items': 0,
                'threads': 0,
                'pending': 0,
                'categories': {},
                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
    def get_action_items(self, status_filter: str = "All") -> List[Dict[str, Any]]:
        try:
            # Expanded search query to catch more action items
            action_keywords = [
                "action", "todo", "task", "urgent", "required", "asap",
                "important", "priority", "critical", "deadline",
                "please", "kindly", "review", "approve", "confirm",
                "due", "by", "eod", "today", "tomorrow",
                "complete", "update", "send", "submit", "prepare"
            ]
            
            # Build the search query - make it more inclusive
            action_query = "("
            action_query += " OR ".join([f"subject:{keyword}" for keyword in action_keywords])
            action_query += ") OR ("
            action_query += " OR ".join([f"body:{keyword}" for keyword in action_keywords])
            action_query += ")"
            
            # Add status filters
            if status_filter.lower() == "pending":
                action_query += " AND is:unread"
            elif status_filter.lower() == "completed":
                action_query += " AND is:read"

            # Execute the search with increased results
            results = self.service.users().messages().list(
                userId='me',
                q=action_query,
                maxResults=100  # Increased from 50
            ).execute()
            
            messages = results.get('messages', [])
            action_items = []
            
            for msg in messages:
                try:
                    email_data = self.service.users().messages().get(
                        userId='me',
                        id=msg['id'],
                        format='full'
                    ).execute()
                    
                    # Extract content using existing method
                    content = self._extract_email_content(email_data)
                    
                    # Parse email data
                    headers = {h['name']: h['value'] for h in email_data['payload']['headers']}
                    subject = headers.get('Subject', '(No subject)')
                    sender = headers.get('From', '(Unknown sender)').replace('<', '').replace('>', '')  # Clean email format
                    date = headers.get('Date', '')
                    
                    # Extract body with better multipart handling
                    body = content.get('body', '')
                    if not body:
                        body = email_data.get('snippet', '')
                    
                    # Check if the email actually contains action items
                    if self._contains_action_items(subject, body):
                        action_item = {
                            'id': msg['id'],
                            'subject': subject,
                            'sender': sender,
                            'date': date,
                            'body': body,
                            'status': 'completed' if 'UNREAD' not in email_data['labelIds'] else 'pending',
                            'is_action_item': True,
                            'priority': self._determine_priority(subject, body, email_data['labelIds'], sender),
                            'due_date': self._extract_due_date(subject, body),
                            'assignee': self._extract_assignee(subject, body),
                            'action_summary': self._generate_action_summary(subject, body)
                        }
                        action_items.append(action_item)
                    
                except Exception as e:
                    print(f"Error processing individual action item: {e}")
                    continue
            
            return action_items
            
        except Exception as e:
            print(f"Error getting action items: {e}")
            return []

    def _contains_action_items(self, subject: str, body: str) -> bool:
        """Check if the email contains actual action items."""
        text = f"{subject} {body}".lower()
        
        # Action verbs that indicate tasks
        action_verbs = [
            "review", "complete", "update", "send", "submit", "prepare",
            "schedule", "confirm", "approve", "check", "investigate", "implement",
            "create", "modify", "test", "verify", "analyze", "coordinate"
        ]
        
        # Task-related phrases
        task_phrases = [
            "please", "kindly", "need to", "should", "must", "have to",
            "required", "action needed", "action required", "to-do", "todo",
            "deadline", "due by", "eod", "asap"
        ]
        
        # Check for action verbs
        if any(f" {verb} " in f" {text} " for verb in action_verbs):
            return True
            
        # Check for task phrases
        if any(phrase in text for phrase in task_phrases):
            return True
            
        return False

    def _build_gmail_query(self, query: str) -> str:
        query = query.lower()
        
        # Handle natural language patterns
        patterns = [
            (r"show me emails (from|to|about|regarding|with|related to) (.+)$", r"\2"),
            (r"give me emails (from|to|about|regarding|with|related to) (.+)$", r"\2"),
            (r"emails (from|to|about|regarding|with|related to) (.+)$", r"\2"),
            (r"find emails (from|to|about|regarding|with|related to) (.+)$", r"\2"),
            (r"search emails (from|to|about|regarding|with|related to) (.+)$", r"\2")
        ]
        
        # Try to match patterns
        for pattern, replacement in patterns:
            match = re.search(pattern, query)
            if match:
                query = match.expand(replacement)
                break
        
        # Handle specific fields
        if "from:" in query:
            return query
        if "to:" in query:
            return query
        if "subject:" in query:
            return query
        
        # If no specific pattern matched, treat as a general search term
        # Remove common words that might interfere with Gmail search
        common_words = ["the", "and", "or", "in", "on", "at", "for", "to", "from", "show", "me", "give", "find", "search", "related", "regarding", "about", "with"]
        query_words = [word for word in query.split() if word not in common_words]
        return ' '.join(query_words) if query_words else query
        assignee_patterns = [
            r'assigned\s+to\s+([\w\s]+)',
            r'please\s+([\w\s]+)\s+to',
            r'attention[:\s]+([\w\s]+)',
            r'for\s+([\w\s]+)\s+to\s+review'
        ]
        
        for pattern in assignee_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return ""

    def _generate_action_summary(self, subject: str, body: str) -> str:
        """Generate a concise summary of the action item."""
        try:
            # Combine subject and first few words of body
            text = f"{subject}\n{body[:200]}"
            
            # Use NLP to extract key action phrases
            doc = self.nlp(text)
            action_phrases = []
            
            for sent in doc.sents:
                if any(action_word in sent.text.lower() for action_word in ["please", "need", "must", "should", "review"]):
                    action_phrases.append(sent.text)
                    break
            
            if action_phrases:
                return action_phrases[0]
            return subject
            
        except Exception as e:
            print(f"Error generating action summary: {e}")
            return subject            

    def _determine_priority(self, subject: str, body: str, labels: List[str], sender: str = '') -> str:
        # Convert to lowercase for case-insensitive matching
        subject_lower = subject.lower()
        body_lower = body.lower()
        
        # Check for IMPORTANT label first
        if 'IMPORTANT' in labels:
            return 'High'
        
        # High-priority keywords with weights
        priority_keywords = {
            'urgent': 3.0,
            'asap': 3.0,
            'important': 2.5,
            'priority': 2.5,
            'critical': 2.5,
            'emergency': 3.0,
            'deadline': 2.0,
            'required': 1.5,
            'action': 1.5,
            'review': 1.2,
            'approve': 1.2,
            'immediate': 2.0
        }
        
        # Calculate priority score
        priority_score = 0
        
        # Check subject line with higher weight (2x)
        for keyword, weight in priority_keywords.items():
            if keyword in subject_lower:
                priority_score += weight * 2
        
        # Check body with normal weight
        for keyword, weight in priority_keywords.items():
            if keyword in body_lower:
                priority_score += weight
        
        # Add score for exclamation marks
        priority_score += (subject + body).count('!') * 0.5
        
        # Determine priority based on refined thresholds
        if priority_score >= 4.0:
            return 'High'
        elif priority_score >= 2.0:
            return 'Medium'
        return 'Normal'

    def _determine_category(self, subject: str, body: str, labels: List[str], priority: str) -> str:
        subject_lower = subject.lower()
        body_lower = body.lower()
        
        # Urgent category checks
        urgent_indicators = [
            'urgent', 'asap', 'emergency', 'critical',
            'immediate', 'priority', '!!!!'
        ]
        if (priority == 'High' and 
            (any(indicator in subject_lower for indicator in urgent_indicators) or
             any(indicator in body_lower for indicator in urgent_indicators))):
            return 'Urgent'
        
        # Action Item category checks
        action_indicators = [
            'action', 'todo', 'task', 'review', 'approve',
            'complete', 'update', 'submit', 'prepare',
            'required', 'need', 'must', 'should'
        ]
        if any(indicator in subject_lower for indicator in action_indicators) or \
           any(indicator in body_lower for indicator in action_indicators):
            return 'Action Item'
        
        # Follow-up category checks
        followup_indicators = [
            'follow up', 'followup', 'follow-up',
            'update', 'status', 'progress'
        ]
        if any(indicator in subject_lower for indicator in followup_indicators) or \
           any(indicator in body_lower for indicator in followup_indicators):
            return 'Follow-up'
        
        return 'General'

    def get_emails(self, filter_type: str = "All", sort_by: str = "Date", search: str = None) -> List[Dict[str, Any]]:
        try:
            # Build the query based on filter type and search
            query = "(-from:(*@*.noreply.com) -from:(*@automated.*) -from:(notification@*) -from:(no-reply@*) -from:(noreply@*))"
            
            if search:
                # Handle natural language queries
                search_lower = search.lower()
                if "give me emails from" in search_lower:
                    sender = search_lower.replace("give me emails from", "").strip()
                    query += f" from:{sender}"
                elif "from" in search_lower:
                    sender = search_lower.replace("from", "").strip()
                    query += f" from:{sender}"
                elif "emails from" in search_lower:
                    sender = search_lower.replace("emails from", "").strip()
                    query += f" from:{sender}"
                else:
                    # General search across all fields
                    query += f" {search}"

            # Add filter type to query
            if filter_type == "Important":
                query += " (label:important OR subject:(urgent OR important OR priority OR critical))"
            elif filter_type != "All":
                query += f" label:{filter_type}"

            print(f"[DEBUG] Search query: {query}")  # Debug log
            
            # Get emails from Gmail API with increased results
            results = self.service.users().messages().list(
                userId='me',
                q=query,
                maxResults=100  # Increased from default
            ).execute()
            
            print(f"[DEBUG] Raw Gmail API results: {results}")  # Debug log
            
            messages = results.get('messages', [])
            emails = []
            
            for msg in messages:
                email_data = self.service.users().messages().get(
                    userId='me',
                    id=msg['id'],
                    format='full'
                ).execute()
                
                # Extract headers
                headers = {h['name']: h['value'] for h in email_data['payload']['headers']}
                subject = headers.get('Subject', '(No subject)')
                sender = headers.get('From', '(Unknown sender)')
                date = headers.get('Date', '')
                
                # Extract content with AI analysis
                content_data = self._extract_email_content(email_data)
                body = content_data.get('body', '')
                ai_summary = content_data.get('ai_summary', '')
                ai_response = content_data.get('ai_response', '')
                
                # Determine priority and category
                priority = self._determine_priority(subject, body, email_data['labelIds'])
                category = self._determine_category(subject, body, email_data['labelIds'], priority)
                
                # Create or update email in database
                
                with Session(engine) as session:
                    email_record = session.query(Email).filter(Email.id == msg['id']).first()
                    if not email_record:
                        email_record = Email(
                            id=msg['id'],
                            sender=sender,
                            subject=subject,
                            body=body,
                            timestamp=datetime.strptime(date, '%a, %d %b %Y %H:%M:%S %z'),
                            category=category,
                            summary=ai_summary,
                            ai_response=ai_response
                        )
                        session.add(email_record)
                    else:
                        email_record.summary = ai_summary
                        email_record.ai_response = ai_response
                    session.commit()

                # Create email object for display
                email = {
                    'id': msg['id'],
                    'subject': subject,
                    'sender': sender,
                    'timestamp': date,
                    'body': body,
                    'summary': ai_summary,
                    'ai_response': ai_response,
                    'priority': priority,
                    'category': category
                }
                emails.append(email)
            
            return self._sort_emails(emails, sort_by)
            
        except Exception as e:
            print(f"Error getting emails: {e}")
            return []

    def initialize_vector_store(self):
        try:
            emails = self.fetch_emails(1000)  # Fetch more emails for better search
            self.vector_store.add_emails(emails)
            self.vector_store.persist()  # Make sure to persist after adding emails
            print(f"Successfully initialized vector store with {len(emails)} emails")
        except Exception as e:
            print(f"Error initializing vector store: {e}")

    def _extract_email_content(self, email_data):
        try:
            parts = []
            if 'payload' in email_data:
                if 'parts' in email_data['payload']:
                    for part in email_data['payload']['parts']:
                        if part.get('mimeType', '').startswith('text/html') and 'data' in part.get('body', {}):
                            decoded = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
                            soup = BeautifulSoup(decoded, 'html.parser')
                            # Remove unwanted elements
                            for tag in soup(['script', 'style', 'head', 'title', 'meta', '[document]', 'header', 'footer', 'nav']):
                                tag.decompose()
                            # Remove HTML comments
                            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                                comment.extract()
                            # Get clean text
                            text = soup.get_text(separator=' ', strip=True)
                            text = ' '.join(text.split())
                            # Escape email addresses and HTML characters
                            text = re.sub(r'([\w\.-]+@[\w\.-]+)', lambda m: html.escape(m.group(1)), text)
                            if text:
                                parts.append(text)
                        elif part.get('mimeType', '').startswith('text/plain') and 'data' in part.get('body', {}):
                            decoded = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
                            text = ' '.join(decoded.split())
                            # Escape email addresses
                            text = re.sub(r'([\w\.-]+@[\w\.-]+)', lambda m: html.escape(m.group(1)), text)
                            if text:
                                parts.append(text)
                elif 'body' in email_data['payload'] and 'data' in email_data['payload']['body']:
                    decoded = base64.urlsafe_b64decode(email_data['payload']['body']['data']).decode('utf-8', errors='ignore')
                    soup = BeautifulSoup(decoded, 'html.parser')
                    for tag in soup(['script', 'style', 'head', 'title', 'meta', '[document]', 'header', 'footer', 'nav']):
                        tag.decompose()
                    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                        comment.extract()
                    text = soup.get_text(separator=' ', strip=True)
                    text = ' '.join(text.split())
                    # Escape email addresses
                    text = re.sub(r'([\w\.-]+@[\w\.-]+)', lambda m: html.escape(m.group(1)), text)
                    if text:
                        parts.append(text)

            content = ' '.join(parts)
            # Clean any remaining HTML tags
            content = re.sub(r'<[^>]+>', '', content)
            # Ensure all HTML entities are properly escaped
            content = html.escape(content)
            content = ' '.join(content.split())

            return {
                'body': content if content else "No content available",
                'ai_summary': "",  # Initialize empty
                'ai_response': ""   # Initialize empty
            }
        except Exception as e:
            print(f"Error extracting email content: {e}")
            return {'body': "No content available", 'ai_summary': "", 'ai_response': ""}

    def _get_fallback_summary(self, content: str, subject: str = '', sender: str = '') -> str:
        """Generate a fallback summary when AI summary fails."""
        try:
            if not content:
                return "No content available"
            # Just return the first 100 characters as a fallback summary
            return content[:100] + "..." if len(content) > 100 else content
        except Exception as e:
            print(f"Error extracting fallback summary: {e}")
            return "No summary available"

    def _build_query(self, filter_type: str = "All", search: str = None) -> str:
        query_parts = []
        # Only add exclusion if you want to filter out noise
        # query_parts.append("(-from:(*@*.noreply.com) -from:(*@automated.*) -from:(notification@*) -from:(no-reply@*) -from:(noreply@*))")
        if search:
            query_parts.append(search.strip())
        return ' '.join(query_parts)

    def _build_gmail_query(self, query: str) -> str:
        query = query.lower()
        
        # Handle natural language patterns
        patterns = [
            (r"show me emails (from|to|about|regarding|with|related to) (.+)$", r"\2"),
            (r"give me emails (from|to|about|regarding|with|related to) (.+)$", r"\2"),
            (r"emails (from|to|about|regarding|with|related to) (.+)$", r"\2"),
            (r"find emails (from|to|about|regarding|with|related to) (.+)$", r"\2"),
            (r"search emails (from|to|about|regarding|with|related to) (.+)$", r"\2")
        ]
        
        # Try to match patterns
        for pattern, replacement in patterns:
            match = re.search(pattern, query)
            if match:
                query = match.expand(replacement)
                break
        
        # Handle specific fields
        if "from:" in query:
            return query
        if "to:" in query:
            return query
        if "subject:" in query:
            return query
        
        # If no specific pattern matched, treat as a general search term
        # Remove common words that might interfere with Gmail search
        common_words = ["the", "and", "or", "in", "on", "at", "for", "to", "from", "show", "me", "give", "find", "search", "related", "regarding", "about", "with"]
        query_words = [word for word in query.split() if word not in common_words]
        return ' '.join(query_words) if query_words else query

    def _sort_emails(self, emails: List[Dict[str, Any]], sort_by: str) -> List[Dict[str, Any]]:
        if sort_by == 'Date':
            return sorted(emails, key=lambda x: parsedate_to_datetime(x['timestamp']), reverse=True)
        elif sort_by == 'Sender':
            return sorted(emails, key=lambda x: x['sender'])
        elif sort_by == 'Subject':
            return sorted(emails, key=lambda x: x['subject'])
        return emails

    def train_email_classifier(self):
        try:
            emails = self.fetch_emails(500)
            texts = [email.get('body', '') for email in emails if email.get('body', '').strip()]
            if not texts:
                print("No valid email content found for training")
                return
                
            categories = ['Urgent', 'Meeting', 'Invoice', 'General', 'Follow-up']
            labels = []
            client = openai.OpenAI(api_key="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")
            
            for text in texts:
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": f"Categorize this email into one of: {', '.join(categories)}"},
                            {"role": "user", "content": text}
                        ]
                    )
                    labels.append(response.choices[0].message.content.strip())
                except Exception as e:
                    print(f"Error categorizing email: {e}")
                    labels.append('General')
                    
            if len(texts) == len(labels) and texts:
                self.vectorizer.fit(texts)
                features = self.vectorizer.transform(texts)
                self.classifier.fit(features, labels)
                print(f"Successfully trained classifier with {len(texts)} emails")
            else:
                print("Insufficient data for training")
        except Exception as e:
            print(f"Error training classifier: {e}")

    def _generate_ai_response_local(self, prompt: str) -> str:
        try:
            response = self.local_model(prompt, max_length=200, min_length=50, do_sample=False)
            return response[0]['summary_text']
        except Exception as e:
            print(f"Error in local model: {e}")
            return "Error generating response locally."

    def _get_fallback_response(self, prompt: str) -> str:
        """Provide a fallback response when AI generation fails."""
        if "summary" in prompt.lower():
            return "No summary available due to service unavailability."
        elif "category" in prompt.lower():
            return "General"
        elif "priority" in prompt.lower():
            return "Medium"
        else:
            return "AI analysis not available at the moment."

    def toggle_star(self, email_id: str) -> None:
        try:
            # Get current labels
            msg = self.service.users().messages().get(userId='me', id=email_id).execute()
            current_labels = msg['labelIds']
            
            # Toggle STARRED label
            if 'STARRED' in current_labels:
                current_labels.remove('STARRED')
            else:
                current_labels.append('STARRED')
            
            # Update labels
            self.service.users().messages().modify(
                userId='me',
                id=email_id,
                body={'removeLabelIds': ['STARRED'] if 'STARRED' in msg['labelIds'] else [],
                      'addLabelIds': ['STARRED'] if 'STARRED' not in msg['labelIds'] else []}
            ).execute()
            
            print(f"Successfully toggled star for email {email_id}")
        except Exception as e:
            print(f"Error toggling star: {e}")
    
    def mark_email_complete(self, email_id: str) -> None:
        try:
            # Mark as read and update in database
            self.service.users().messages().modify(
                userId='me',
                id=email_id,
                body={'removeLabelIds': ['UNREAD']}
            ).execute()
            
            # Update status in database
            session = SessionLocal()
            try:
                email = session.query(Email).filter(Email.id == email_id).first()
                if email:
                    email.status = 'completed'
                    session.commit()
            finally:
                session.close()
                
            print(f"Successfully marked email {email_id} as complete")
        except Exception as e:
            print(f"Error marking email as complete: {e}")

    def update_action_item(self, action_item_id: str, status: str) -> None:
        try:
            session = SessionLocal()
            action_item = session.query(Email).filter(Email.id == action_item_id).first()
            if action_item:
                action_item.status = status
                session.commit()
                print(f"Action item {action_item_id} updated to status: {status}")
            else:
                print(f"Action item {action_item_id} not found.")
        except Exception as e:
            print(f"Error updating action item: {e}")
        finally:
            session.close()

    def _parse_email_data(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse email data into a structured format."""
        try:
            headers = {h['name']: h['value'] for h in email_data['payload']['headers']}
            snippet = email_data.get('snippet', '')
            subject = headers.get('Subject', '(No subject)')
            sender = headers.get('From', '(Unknown sender)')
            timestamp = headers.get('Date', '')

            return {
                'id': email_data['id'],
                'subject': subject,
                'sender': sender,
                'timestamp': timestamp,
                'body': snippet,
                'is_unread': 'UNREAD' in email_data.get('labelIds', []),
                'is_important': 'IMPORTANT' in email_data.get('labelIds', []),
                'is_starred': 'STARRED' in email_data.get('labelIds', []),
            }
        except Exception as e:
            print(f"Error parsing email data: {e}")
            return {}

    def categorize_email(self, email: Dict[str, Any]) -> List[str]:
        """Categorize email using trained ML model or fallback to rule-based."""
        try:
            text = f"{email.get('subject', '')} {email.get('body', '')}"
            features = self.vectorizer.transform([text])
            prediction = self.classifier.predict(features)
            return [prediction[0]]
        except Exception as e:
            print(f"ML failed, using fallback rules: {e}")
        
        # Rule-based fallback
        categories = []
        subject = email.get('subject', '').lower()
        body = email.get('body', '').lower()
        if any(word in subject or word in body for word in ['urgent', 'asap', 'emergency']):
            categories.append('Urgent')
        if any(word in subject or word in body for word in ['meeting', 'call', 'conference']):
            categories.append('Meeting')
        if any(word in subject or word in body for word in ['invoice', 'payment', 'bill']):
            categories.append('Invoice')
        if any(word in subject or word in body for word in ['follow', 'update', 'status']):
            categories.append('Follow-up')
        if not categories:
            categories.append('General')
        return list(set(categories))

    def _preprocess_search_query(self, query: str) -> str:
        """Preprocess the search query using spaCy"""
        doc = self.nlp(query.lower())
        # Extract important tokens and their lemmas
        search_terms = []
    
        for token in doc:
            if not token.is_stop and not token.is_punct:
                search_terms.append(token.lemma_)
        return ' '.join(search_terms)

    def _sort_emails(self, emails: List[Dict[str, Any]], sort_by: str = "Date") -> List[Dict[str, Any]]:
        """Sort emails based on specified criteria"""
        try:
            if sort_by == "Date":
                return sorted(emails, key=lambda x: parsedate_to_datetime(x['timestamp']), reverse=True)
            elif sort_by == "Priority":
                priority_order = {"High": 0, "Normal": 1, "Low": 2}
                return sorted(emails, key=lambda x: priority_order.get(x.get('priority', 'Low'), 2))
            return emails
        except Exception as e:
            print(f"Error sorting emails: {e}")
            return emails

    def get_todo_emails(self) -> list:
        todo_patterns = [
        r"^\s*[-*]\s+.+",  # bullet points
        r"^\s*\d+\.\s+.+",  # numbered lists
        r"\bplease\b", r"\bkindly\b", r"\baction required\b", r"\bto do\b",
        r"\bremind\b", r"\bfollow up\b", r"\bcomplete\b", r"\bpending\b"
    ]
        """
        Returns emails that likely contain to-do items or actionable lists.
        """
        try:
            emails = self.fetch_emails(limit=10)  # or whatever limit you want
            todo_emails = []
            todo_patterns = [
                r"^\s*[-*]\s+.+",  # bullet points
                r"^\s*\d+\.\s+.+",  # numbered lists
                r"\bplease\b", r"\bkindly\b", r"\baction required\b", r"\bto do\b",
                r"\bremind\b", r"\bfollow up\b", r"\bcomplete\b", r"\bpending\b"
            ]
            
            for email in emails:
                body = email.get('body', '').lower()
                found = False
                # Check for bullet/numbered lists
                for line in body.splitlines():
                    if re.match(todo_patterns[0], line) or re.match(todo_patterns[1], line):
                        found = True
                    # Check for keyword matches
                    for pattern in todo_patterns[2:]:
                        if re.search(pattern, line):
                            found = True
                            break
                if found:
                    todo_emails.append(email)
            return todo_emails
        except Exception as e:
            print(f"Error in get_todo_emails: {e}")
            return []

    def search_emails(self, search_query: str = None) -> List[Dict[str, Any]]:
        """
        Dynamic smart search: Accepts any query (single word, phrase, or natural language)
        and returns matching emails using Gmail API's search.
        """
       

        def build_dynamic_query(query: str) -> str:
            query = (query or "").lower().strip()
            # Try to extract intent
            from_match = re.search(r"(?:from|by)\s+([a-zA-Z0-9 ._-]+)", query)
            to_match = re.search(r"(?:to)\s+([a-zA-Z0-9 ._-]+)", query)
            subject_match = re.search(r"(?:subject|about|regarding)\s+([a-zA-Z0-9 ._-]+)", query)
            if from_match:
                return f"from:{from_match.group(1).strip()}"
            if to_match:
                return f"to:{to_match.group(1).strip()}"
            if subject_match:
                return f"subject:{subject_match.group(1).strip()}"
            # Remove filler words for general keyword search
            filler = {"show", "me", "emails", "email", "find", "all", "the", "please", "about", "regarding"}
            keywords = [w for w in query.split() if w not in filler]
            return " ".join(keywords) if keywords else query

        try:
            gmail_query = build_dynamic_query(search_query or "")
            print(f"[DEBUG] Gmail API search query: {gmail_query}")

            results = self.service.users().messages().list(
                userId='me',
                q=gmail_query,
                maxResults=100
            ).execute()

            messages = results.get('messages', [])
            if not messages:
                print(f"[DEBUG] No messages found for query: {gmail_query}")
                return []

            formatted_emails = []
            for msg in messages:
                try:
                    email = self.service.users().messages().get(
                        userId='me',
                        id=msg['id'],
                        format='full'
                    ).execute()

                    # Extract headers
                    headers = {h['name']: h['value'] for h in email['payload']['headers']}
                    subject = headers.get('Subject', '(No subject)')
                    sender = headers.get('From', '(Unknown sender)')
                    date = headers.get('Date', '')

                    # Extract content
                    content_data = self._extract_email_content(email)
                    body = content_data.get('body', '')
                    summary = content_data.get('ai_summary', '')

                    # Add more metadata
                    formatted_email = {
                        'id': email['id'],
                        'subject': subject,
                        'sender': sender,
                        'timestamp': date,
                        'body': body or "No content available",
                        'ai_summary': content_data.get('ai_summary', 'No summary available'),
                        'ai_response': content_data.get('ai_response', 'No AI response available'),
                        'is_unread': 'UNREAD' in email.get('labelIds', []),
                        'is_important': 'IMPORTANT' in email.get('labelIds', []),
                        'is_starred': 'STARRED' in email.get('labelIds', []),
                        'has_attachments': any(part.get('filename') for part in email['payload'].get('parts', []))
                                        }

                    formatted_emails.append(formatted_email)
                    print(f"[DEBUG] Added email: {subject}")
                except Exception as e:
                    print(f"Error processing email {msg['id']}: {e}")
                    continue

            print(f"[DEBUG] Successfully processed {len(formatted_emails)} emails")
            return formatted_emails

        except Exception as e:
            print(f"Error in search_emails: {e}")
            return []

    def get_todo_emails(self) -> list:
        """
        Returns emails that likely contain to-do items or actionable lists.
        """
        try:
            emails = self.fetch_emails(limit=10)  # or whatever limit you want
            todo_emails = []
            todo_patterns = [
                r"^\s*[-*]\s+.+",  # bullet points
                r"^\s*\d+\.\s+.+",  # numbered lists
                r"\bplease\b", r"\bkindly\b", r"\baction required\b", r"\bto do\b",
                r"\bremind\b", r"\bfollow up\b", r"\bcomplete\b", r"\bpending\b", r"\bfeedback\b"
            ]
            
            for email in emails:
                body = email.get('body', '').lower()
                found = False
                
                # Check for bullet/numbered lists
                for line in body.splitlines():
                    if re.match(todo_patterns[0], line) or re.match(todo_patterns[1], line):
                        found = True
                        break
                
                # Check for action phrases
                if not found and any(re.search(pat, body) for pat in todo_patterns[2:]):
                    found = True
                
                if found:
                    todo_emails.append(email)
            
            return todo_emails
        except Exception as e:
            print(f"Error in get_todo_emails: {e}")
            return []
