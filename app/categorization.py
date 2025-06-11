from typing import Dict, Any, List

def categorize_email(email: dict) -> list:
    """Categorize email based on its content."""
    categories = []
    subject = email.get('subject', '').lower()
    body = email.get('body', '').lower()

    # Priority-based categorization
    urgent_keywords = ['urgent', 'asap', 'emergency', 'immediate', 'critical']
    if any(keyword in subject for keyword in urgent_keywords):
        categories.append('Urgent')
    elif any(keyword in body for keyword in urgent_keywords):
        categories.append('Urgent')
        
    # Other categories
    if any(keyword in subject or keyword in body for keyword in ['meeting', 'call', 'conference']):
        categories.append('Meeting')
    if any(keyword in subject or keyword in body for keyword in ['invoice', 'payment', 'bill']):
        categories.append('Invoice')
    if any(keyword in subject or keyword in body for keyword in ['follow', 'update', 'status']):
        categories.append('Follow-up')
    
    # Important category based on keywords
    important_keywords = ['important', 'priority', 'attention', 'required', 'urgent', 'asap', 
                        'critical', 'essential', 'crucial', 'vital', 'key', 'significant',
                        'deadline', 'action', 'response needed', 'please respond']
    if any(keyword in subject or keyword in body for keyword in important_keywords):
        if 'Urgent' not in categories:
            categories.append('Important')
    
    # Check for other importance indicators
    if email.get('priority', '').lower() == 'high' or any(label.lower().endswith('important') for label in email.get('labels', [])):
        if 'Urgent' not in categories and 'Important' not in categories:
            categories.append('Important')
    
    if not categories:
        categories.append('General')

    return list(set(categories))  # Remove any duplicates