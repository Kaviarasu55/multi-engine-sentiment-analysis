import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def predict_groq(text):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""You are a strict sentiment classifier. You must respond with exactly one word only: positive, negative, or neutral. No explanation, no punctuation, just one word.

Examples:
Text: "the flight was delayed 4 hours and staff was rude"
Sentiment: negative

Text: "amazing service and very comfortable seats"  
Sentiment: positive

Text: "the flight was okay nothing special"
Sentiment: neutral

Text: "absolutely terrible experience never flying again"
Sentiment: negative

Text: "worst airline i have ever used completely disgusted"
Sentiment: negative

Now classify:
Text: "{text}"
Sentiment:"""
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0
    )

    label = response.choices[0].message.content.strip().lower()

    if label not in ["positive", "negative", "neutral"]:
        label = "neutral"

    return label