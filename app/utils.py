from email.utils import parsedate_to_datetime

def extract_email_content(email_data):
    try:
        content = email_data.get('snippet', '')
        if 'payload' in email_data:
            headers = {h['name']: h['value'] for h in email_data['payload'].get('headers', [])}
            subject = headers.get('Subject', '')
            if subject:
                content = f"Subject: {subject}\n\n{content}"
        return content
    except Exception as e:
        print(f"Error extracting email content: {e}")
        return email_data.get('snippet', '')

def build_query(filter_type: str = "All", search: str = None):
    query_parts = []

    # Handle filter type
    if filter_type == "Unread":
        query_parts.append("is:unread")
    elif filter_type == "Important":
        query_parts.append("is:important")
    elif filter_type == "Starred":
        query_parts.append("is:starred")

    if search:
        # Clean and normalize the search query
        search = search.lower().strip()
        
        # Handle natural language patterns
        if "from:" not in search and ("from" in search or "by" in search or "sent by" in search):
            words = search.split()
            for i, word in enumerate(words):
                if word in ["from", "by"] and i + 1 < len(words):
                    search = search.replace(f"{word} {words[i+1]}", f"from:{words[i+1]}")
                elif word == "sent" and i + 2 < len(words) and words[i+1] == "by":
                    search = search.replace(f"sent by {words[i+2]}", f"from:{words[i+2]}")

        # Handle date-related queries
        date_patterns = {
            "today": "newer_than:1d",
            "yesterday": "newer_than:2d older_than:1d",
            "this week": "newer_than:7d",
            "this month": "newer_than:30d",
            "last week": "newer_than:14d older_than:7d",
            "last month": "newer_than:60d older_than:30d"
        }
        for pattern, replacement in date_patterns.items():
            if pattern in search:
                search = search.replace(pattern, replacement)

        # Handle subject and content queries
        search_terms = search.split()
        keyword_parts = []
        
        for term in search_terms:
            if term.startswith("from:") or term.startswith("newer_than:") or term.startswith("older_than:"):
                query_parts.append(term)
            elif not term.startswith("in:") and not term in ["and", "or", "the", "a", "an"]:
                keyword_parts.append(f"(subject:{term} OR body:{term})")

        if keyword_parts:
            query_parts.append(f"({' OR '.join(keyword_parts)})")

    return " AND ".join(query_parts) if query_parts else ""
