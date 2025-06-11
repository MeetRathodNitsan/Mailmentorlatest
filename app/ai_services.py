import email
import re
from click import prompt
import openai
import requests
import streamlit as st
from app.config import OPENAI_API_KEY, USE_OPENAI_API, LOCAL_MODELS, MODEL_PARAMS
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import requests

# Load the summarization model using a compatible model
summarizer = None
responder = None

def initialize_models():
    global summarizer, responder
    try:
        if summarizer is None:
            # Replace the large BART model with a smaller alternative
            summarizer = pipeline("summarization", model="facebook/bart-base")
        if responder is None:
            responder = pipeline("text2text-generation", model="facebook/blenderbot-400M-distill")
    except Exception as e:
        print(f"Error initializing models: {e}")
        raise

def generate_ai_response(prompt: str) -> str:
    initialize_models()
    if USE_OPENAI_API:
        try:
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional assistant. Provide a single-line response acknowledging the meeting or task. Do not repeat the email content, time, or subject."}, 
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=30
            )
            result = response.choices[0].message.content.strip()
            result = result.split('\n')[0].strip()
            if len(result) > 100:
                result = result[:97] + '...'
            return result
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Error generating response"
    else:
        return generate_local_summary(prompt)

def analyze_email_content(subject: str, content: str) -> tuple[str, str]:
    initialize_models()
    try:
        summary_prompt = f"Summarize this email briefly:\nSubject: {subject}\nContent: {content}"
        suggestion_prompt = f"Suggest a brief action or response for this email:\nSubject: {subject}\nContent: {content}"
        
        if USE_OPENAI_API:
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            # Get summary
            summary_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Provide a one-sentence summary of the key points."}, 
                    {"role": "user", "content": summary_prompt}
                ],
                temperature=0.3,
                max_tokens=50
            )
            summary = summary_response.choices[0].message.content.strip()
            
            # Get suggestion response
            suggestion_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Provide a one-line suggestion or action item."}, 
                    {"role": "user", "content": suggestion_prompt}
                ],
                temperature=0.3,
                max_tokens=30
            )
            suggestion = suggestion_response.choices[0].message.content.strip()
            
            return _post_process_summary(content, summary), suggestion.split('\n')[0].strip()

        # Local model fallback
        summary = generate_local_summary(content)
        suggestion = generate_local_response(subject, content)
        return _post_process_summary(content, summary), suggestion

    except Exception as e:
        print(f"Error generating AI content: {e}")
        return "Error generating summary.", "Error generating suggestion."

def _post_process_summary(original: str, summary: str) -> str:
    """Post-process the AI-generated summary."""
    # Remove any quoted content or timestamps
    summary = summary.split('Content:')[-1].split('Write a professional response:')[0].strip()
    summary = summary.split('Subject:')[0].strip()
    
    # Keep only the first sentence if multiple exist
    sentences = summary.split('.')
    if sentences:
        summary = sentences[0].strip() + '.'
    
    return summary

def generate_local_summary(content: str) -> str:
    try:
        # Truncate content to fit within model's maximum token limit
        max_chars = 512  # Approximate character limit for 128 tokens
        truncated_content = content[:max_chars] if len(content) > max_chars else content
        
        # Use the bart-large-cnn model for summarization
        summary = summarizer(truncated_content, 
                           max_length=MODEL_PARAMS["max_length"], 
                           min_length=MODEL_PARAMS["min_length"], 
                           do_sample=True,
                           temperature=MODEL_PARAMS["temperature"])[0]['summary_text']
        return _post_process_summary(content, summary)
    except Exception as e:
        print(f"Error generating local summary: {e}")
        return "Error generating summary."

def generate_local_response(subject: str, content: str) -> str:
    try:
        # Truncate content to fit within model's maximum token limit
        max_chars = 512  # Approximate character limit for 128 tokens
        truncated_content = content[:max_chars] if len(content) > max_chars else content
        
        prompt = f"Subject: {subject}\nContent: {truncated_content}\nWrite a professional response:"
        response = responder(prompt, 
                           max_length=MODEL_PARAMS["max_length"], 
                           do_sample=True, 
                           temperature=MODEL_PARAMS["temperature"],
                           top_p=MODEL_PARAMS["top_p"])[0]['generated_text']
        return response.split('\n')[0].strip()
    except Exception as e:
        print(f"Error generating local response: {e}")
        return "Error generating response."

def _post_process_summary(original: str, summary: str) -> str:
    # Remove code blocks and inline code
    summary = re.sub(r'```[\s\S]*?```', '', summary)
    summary = re.sub(r'`[^`]*`', '', summary)
    
    # Clean up any remaining markdown or special characters
    summary = re.sub(r'\[.*?\]\(.*?\)', '', summary)  # Remove links
    summary = re.sub(r'[#*_~]', '', summary)  # Remove markdown characters
    
    # Remove any quoted content or timestamps
    summary = summary.split('Content:')[-1].split('Write a professional response:')[0].strip()
    summary = summary.split('Subject:')[0].strip()
    
    # Keep only the first sentence if multiple exist
    sentences = summary.split('.')
    if sentences:
        summary = sentences[0].strip() + '.'
    
    return summary

def generate_ai_response_to_email(subject: str, content: str) -> str:
    """Generate a single, concise response to an email."""
    # Check if email is too short (less than 20 characters)
    if len(content.strip()) < 20:
        return "No need for AI response"
        
    try:
        prompt = f"""Generate a single, professional response to this email:
Subject: {subject}
Content: {content}

Keep the response concise and professional. Respond in exactly one sentence."""
        
        if USE_OPENAI_API:
            response = generate_ai_response(prompt)
            # Ensure single response by taking first sentence and removing any newlines
            response = response.replace('\n', ' ').strip()
            first_sentence = response.split('.')[0].strip()
            return first_sentence + '.' if not first_sentence.endswith('.') else first_sentence
        else:
            response = generate_local_response(subject, content)
            # Apply same processing for local model response
            response = response.replace('\n', ' ').strip()
            first_sentence = response.split('.')[0].strip()
            return first_sentence + '.' if not first_sentence.endswith('.') else first_sentence
            
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Thank you for your email."
    if USE_OPENAI_API:
        try:
            prompt = f"""
            Based on this email, provide TWO concise response options:
            Subject: {subject}
            Content: {content}
            
            Format the response as:
            Option 1: [Short, direct response]
            Option 2: [Alternative response]
            
            Keep each option under 50 words and focus on actionable replies.
            """
            response = generate_ai_response(prompt)
            return response
        except Exception as e:
            print(f"Error generating AI response: {e}")
            return "Option 1: Thank you for your email. I'll review and respond shortly.\nOption 2: I acknowledge your message and will address it soon."
    else:
        # Structured local response
        try:
            response = generate_local_response(subject, content)
            return f"Option 1: {response}\nOption 2: Thank you for your message. I'll review and respond accordingly."
        except Exception as e:
            print(f"Error generating local response: {e}")
            return "Option 1: Thank you for your email. I'll review and respond shortly.\nOption 2: I acknowledge your message and will address it soon."

def parse_email_data(email_data: dict) -> dict:
    """
    Parse email data and generate AI summary and response.
    """
    try:
        # Check if email_data is valid
        if not email_data or not isinstance(email_data, dict):
            print("[DEBUG] Invalid email data format")
            return {}

        # Extract headers safely
        headers = {}
        if 'payload' in email_data and 'headers' in email_data['payload']:
            headers = {h['name']: h['value'] for h in email_data['payload']['headers']}
        
        subject = headers.get('Subject', '(No subject)')
        sender = headers.get('From', 'Unknown')
        content = email_data.get('snippet', '')

        # Add debug logging
        print(f"[DEBUG] Processing email - Subject: {subject}, From: {sender}")

        # Generate AI summary
        summary = analyze_email_content(subject, content)

        # Generate AI response
        ai_response = generate_ai_response_to_email(subject, content)

        return {
            'id': email_data.get('id', ''),
            'subject': subject,
            'sender': sender,
            'content': content,
            'summary': summary,
            'ai_response': ai_response,
            'status': 'pending'
        }
    except Exception as e:
        print(f"[DEBUG] Error in email processing: {str(e)}")
        return {}

def fetch_email_data() -> dict:
    """
    Simulate fetching email data from Gmail.
    Replace this with actual Gmail API integration.
    """
    return {
        "id": "12345",
        "payload": {
            "headers": [
                {"name": "Subject", "value": "Meeting Reminder"},
                {"name": "From", "value": "team@example.com"}
            ]
        },
        "snippet": "This is a reminder for the team meeting scheduled tomorrow at 10 AM."
    }

def preprocess_email_content(content: str) -> str:
    """
    Preprocess email content to remove unnecessary details.
    """
    # Example: Remove signatures or disclaimers
    lines = content.splitlines()
    filtered_lines = [line for line in lines if not line.strip().startswith("--")]
    return "\n".join(filtered_lines)

if __name__ == "__main__":
    # Fetch email data
    email_data = fetch_email_data()

    # Parse the email data
    parsed_data = parse_email_data(email_data)

    # Display the email details
    st.write(f"**Subject:** {parsed_data['subject']}")
    st.write(f"**Sender:** {parsed_data['sender']}")
    st.write(f"**Content:** {parsed_data['content']}")
    st.write(f"**AI Summary:** {parsed_data['summary']}")
    st.write(f"**AI Response:** {parsed_data['ai_response']}")