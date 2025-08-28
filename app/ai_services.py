from transformers import pipeline
import torch

# Global variables to hold the models
summarizer = None


# You can add other models here, e.g., classifier = None

def initialize_models():
    """
    Initializes all AI models once when the application starts.
    This prevents the 'Already borrowed' error by not reloading models in loops.
    """
    global summarizer
    if summarizer is None:
        print("Loading NLP models...")
        try:
            # Use GPU if available, otherwise CPU
            device = 0 if torch.cuda.is_available() else -1
            summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=device
            )
            print("NLP models loaded successfully.")
        except Exception as e:
            print(f"Error loading NLP models: {e}")
            # Handle model loading failure gracefully
            summarizer = None


def analyze_email_content(subject: str, body: str) -> tuple[str, str]:
    """
    Analyzes email content to generate a summary and a suggested response.
    Uses the globally loaded summarizer model.
    """
    if summarizer is None:
        print("Summarizer model is not available.")
        # Return the subject as a fallback summary and a generic response
        return subject, "Could not generate AI response."

    # --- Generate Summary ---
    summary = subject  # Default summary is the subject
    try:
        # Only generate a new summary for emails with a reasonably long body
        if len(body.split()) > 40:
            text_to_summarize = f"Subject: {subject}\n\n{body}"

            # Truncate the input text if it's too long for the model
            max_model_length = 1024

            # Use the model's tokenizer to be precise about token length
            tokenized_input = summarizer.tokenizer.encode(text_to_summarize, truncation=False)
            if len(tokenized_input) > max_model_length:
                # Decode the truncated tokens back to a string
                truncated_ids = tokenized_input[:max_model_length]
                text_to_summarize = summarizer.tokenizer.decode(truncated_ids, skip_special_tokens=True)

            # FIX: Use simpler, safer length constraints for the summary output.
            # This avoids edge cases where min_length could be too close to max_length,
            # which was causing the 'index out of range' error.
            max_len = 150  # A reasonable maximum summary length
            min_len = 30  # A reasonable minimum summary length

            # Generate the summary
            summary_result = summarizer(
                text_to_summarize,
                max_length=max_len,
                min_length=min_len,
                do_sample=False
            )
            summary = summary_result[0]['summary_text']
    except Exception as e:
        print(f"Error during summarization: {e}")
        # Fallback to subject if summarization fails

    # --- Generate Suggested Response (Placeholder) ---
    suggested_response = "Thank you for your email. I will get back to you shortly."
    if "action" in summary.lower() or "task" in summary.lower():
        suggested_response = "Thank you for the update. I will take care of this action item."
    elif "?" in body:
        suggested_response = "Thanks for your question. Let me look into that and I'll get back to you."

    return summary, suggested_response
