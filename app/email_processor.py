import base64
import email
from email import message_from_bytes
from googleapiclient.discovery import build
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, Any
from datetime import datetime, timezone
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import func
import re
from app.config import SessionLocal
from app.models import Email
from app.categorization import categorize_email  # Assuming categorization.py is in 'app' folder

# Set up logging
logger = logging.getLogger(__name__)


class EmailProcessor:
    def __init__(self, credentials):
        """
        Initializes the EmailProcessor with user credentials and required models.
        """
        self.credentials = credentials
        self.service = build('gmail', 'v1', credentials=self.credentials)

        logger.info("Loading SentenceTransformer model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Model loaded successfully.")

    def fetch_and_save_emails(self, limit=50) -> List[Dict[str, Any]]:
        """
        Fetches emails from Gmail, processes them, categorizes them, and saves to the database.
        Returns a list of emails.
        """
        db_session = SessionLocal()
        fetched_emails = []
        try:
            # Fetch email IDs from Gmail, excluding drafts and chats
            results = self.service.users().messages().list(
                userId='me', maxResults=limit, q='-in:drafts -in:chats'
            ).execute()
            messages = results.get('messages', [])

            if not messages:
                logger.info("No new emails found.")
                return []

            for msg_info in messages:
                # Check if email already exists in the database
                existing_email = db_session.query(Email).filter(Email.id == msg_info['id']).first()
                if existing_email:
                    logger.info(f"Email with ID {msg_info['id']} already exists. Skipping.")
                    continue

                # Fetch the full email content
                msg = self.service.users().messages().get(userId='me', id=msg_info['id'], format='raw').execute()
                raw_email = base64.urlsafe_b64decode(msg['raw'].encode('ASCII'))
                email_message = message_from_bytes(raw_email)

                # Extract and decode email parts
                subject = self.decode_header(email_message['subject'])
                sender = self.decode_header(email_message['from'])
                body = self.get_email_body(email_message)

                # Extract recipient
                recipient_header = email_message.get('to')
                recipient = self.decode_header(recipient_header) if recipient_header else 'me'

                # Parse the timestamp
                try:
                    timestamp_str = email_message['date']
                    timestamp_dt = datetime.strptime(timestamp_str, '%a, %d %b %Y %H:%M:%S %z').astimezone(timezone.utc)
                except Exception:
                    timestamp_dt = datetime.now(timezone.utc)

                # Categorize the email
                try:
                    categories = categorize_email({"subject": subject, "body": body}) or []
                except Exception as e:
                    logger.error(f"Categorization failed: {e}")
                    categories = []

                # Normalize categories
                categories = [c.strip().title() for c in categories if c.strip()]
                if not categories:
                    categories = ["Uncategorized"]

                category_str = ", ".join(sorted(set(categories)))

                # Create Email object
                new_email = Email(
                    id=msg['id'],
                    sender=sender,
                    recipient=recipient,
                    subject=subject,
                    body=body,
                    timestamp=timestamp_dt,
                    category=category_str,
                    status='unread' if 'UNREAD' in msg.get('labelIds', []) else 'read'
                )
                db_session.add(new_email)
                fetched_emails.append(new_email)

            db_session.commit()
            logger.info(f"Successfully fetched and saved {len(fetched_emails)} new emails.")

        except SQLAlchemyError as e:
            db_session.rollback()
            logger.error(f"Database error during email ingestion: {e}")
        except Exception as e:
            logger.error(f"An error occurred while fetching emails: {e}")
        finally:
            db_session.close()

        return fetched_emails

    def get_email_stats(self) -> Dict[str, Any]:
        """
        Dynamically analyzes emails and returns key statistics for the dashboard.
        """
        logger.info("Fetching email statistics from Gmail API and database...")
        stats = {}
        db_session = SessionLocal()

        try:
            # 1. Gmail unread count
            unread_results = self.service.users().labels().get(userId='me', id='UNREAD').execute()
            stats['unread'] = unread_results.get('messagesUnread', 0)

            # 2. Gmail thread count
            threads_results = self.service.users().threads().list(userId='me').execute()
            stats['threads'] = len(threads_results.get('threads', []))

            # 3. Count action items
            action_items_count = db_session.query(Email).filter(Email.category.ilike('%Action Item%')).count()
            stats['action_items'] = action_items_count

            # 4. Count categories (split multi-categories)
            category_rows = db_session.query(Email.category).all()
            flat_counts = {}
            for (cat_str,) in category_rows:
                if not cat_str:
                    cat_str = "Uncategorized"
                for cat in [c.strip() for c in cat_str.split(",")]:
                    if cat:
                        flat_counts[cat] = flat_counts.get(cat, 0) + 1

            if not flat_counts:
                flat_counts = {"Uncategorized": 0}
            stats['categories'] = flat_counts

            # 5. Pending TODOs
            stats['pending'] = db_session.query(Email).filter(Email.status == "todo").count()

        except Exception as e:
            logger.error(f"An error occurred while getting email stats: {e}")
            stats = {
                'unread': 'N/A',
                'action_items': 'N/A',
                'threads': 'N/A',
                'pending': 'N/A',
                'categories': {},
                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        finally:
            db_session.close()

        stats['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return stats

    # --- Helpers ---
    def decode_header(self, header):
        if header is None:
            return ""
        decoded_parts = []
        for part, encoding in email.header.decode_header(header):
            if isinstance(part, bytes):
                decoded_parts.append(part.decode(encoding or 'utf-8', errors='ignore'))
            else:
                decoded_parts.append(part)
        return "".join(decoded_parts)

    def get_email_body(self, msg):
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == 'text/plain' and 'attachment' not in part.get('Content-Disposition', ''):
                    body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    break
        else:
            if msg.get_content_type() == 'text/plain':
                body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
        return body

    def get_action_items(self, status_filter: str = "All", limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        logger.info(f"Fetching action items with filter: {status_filter}...")
        session = SessionLocal()
        try:
            query = (
                session.query(Email)
                .filter(Email.category.ilike("%Action Item%"))
                .order_by(Email.timestamp.desc())
            )
            if status_filter != "All":
                query = query.filter(Email.status.ilike(status_filter))
            emails = query.offset(offset).limit(limit).all()
            return [
                {
                    "id": e.id,
                    "subject": e.subject,
                    "sender": e.sender,
                    "recipient": e.recipient,
                    "date": e.timestamp.strftime("%Y-%m-%d"),
                    "status": e.status,
                    "body": e.body,
                    "summary": getattr(e, "summary", None),
                    "ai_response": getattr(e, "ai_response", None),
                }
                for e in emails
            ]
        except Exception as e:
            logger.error(f"Error fetching action items: {e}")
            return []
        finally:
            session.close()

    def get_todo_emails(self, status_filter: str = "All", limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Fetch TODO emails from the database with optional status filter.
        Extracts deadlines from email content if available.
        """
        session = SessionLocal()
        try:
            query = (
                session.query(Email)
                .filter(Email.status == "todo")
                .order_by(Email.timestamp.desc())
            )

            # Apply status filter if not "All"
            if status_filter != "All":
                query = query.filter(Email.status.ilike(status_filter))

            emails = query.offset(offset).limit(limit).all()

            todo_items = []
            for email_obj in emails:
                # Try extracting a deadline from subject/body (e.g. "due on 2025-09-01" or "Deadline: Sept 5")
                deadline = None
                deadline_match = re.search(
                    r"(?:due|deadline)[:\s]*([A-Za-z0-9 ,/-]+)",
                    f"{email_obj.subject} {email_obj.body}",
                    re.IGNORECASE
                )
                if deadline_match:
                    deadline = deadline_match.group(1).strip()

                todo_items.append({
                    "id": email_obj.id,
                    "subject": email_obj.subject,
                    "sender": email_obj.sender,
                    "recipient": email_obj.recipient,
                    "date": email_obj.timestamp.strftime("%Y-%m-%d"),
                    "status": email_obj.status,
                    "deadline": deadline,
                    "body": email_obj.body,
                    "summary": getattr(email_obj, "summary", None),
                    "ai_response": getattr(email_obj, "ai_response", None)
                })

            return todo_items

        except Exception as e:
            logger.error(f"Error fetching TODO emails: {e}")
            return []
        finally:
            session.close()

