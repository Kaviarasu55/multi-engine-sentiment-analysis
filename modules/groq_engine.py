import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """
You are an expert sentiment classification model.

Your task is to classify the OVERALL sentiment of an airline review.

Return ONLY ONE of these labels:
positive
negative
neutral

Rules:
- positive:
  The overall opinion expresses satisfaction, praise, enjoyment, recommendation,
  appreciation, or a good experience.

- negative:
  The overall opinion expresses complaint, disappointment, frustration,
  delay, poor service, discomfort, damage, bad experience,
  or any negative opinion.

- neutral:
  Use ONLY when the text contains no clear opinion or is purely factual.

Important:
- For mixed sentiment, choose the DOMINANT sentiment.
- Never explain.
- Never justify.
- Never use punctuation.
- Never output anything except:
positive
negative
neutral

Examples:

"The flight was amazing and the crew were friendly."
positive

"The flight was delayed for five hours and the seats were uncomfortable."
negative

"The flight departed from Chennai at 9 PM."
neutral

"The food wasn't great but I enjoyed the journey."
positive

"The flight was delayed but the staff were extremely helpful."
negative

"Seats were damaged but the restrooms were clean."
negative
"""


def predict_groq(text):
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        temperature=0,
        max_tokens=50,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )

    raw_output = response.choices[0].message.content.strip().lower()

    from pprint import pprint
    pprint(response.model_dump())

    if "positive" in raw_output:
        return "positive"
    elif "negative" in raw_output:
        return "negative"
    elif "neutral" in raw_output:
        return "neutral"

    return "neutral"
