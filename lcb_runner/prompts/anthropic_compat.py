try:
    from anthropic import AI_PROMPT, HUMAN_PROMPT
except ImportError:
    HUMAN_PROMPT = "\n\nHuman:"
    AI_PROMPT = "\n\nAssistant:"
