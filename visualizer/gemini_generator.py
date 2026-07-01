import json
import streamlit as st

PROMPT = """You are an expert ML and mathematics tutor helping students solve Deep-ML coding challenges.

Given a concept name, generate a complete explanation with a concrete worked example using real numbers.

Return ONLY a valid JSON object — no markdown, no text outside the JSON:

{{
  "description": "One clear sentence explaining what this concept computes and why it matters",
  "difficulty": "Easy",
  "category": "Linear Algebra or ML Algorithms or Metrics or Activation Functions or Optimization or Statistics or Deep Learning",
  "inputs": {{
    "input_name": "actual value (number, list, or matrix as nested list)"
  }},
  "steps": [
    {{
      "title": "Step title",
      "explanation": "Explanation with actual computed numbers from the example — show the arithmetic",
      "math": "Simple LaTeX formula — KaTeX compatible, no \\\\begin or \\\\end blocks",
      "result": "The actual computed result as a string",
      "code": "1-3 lines of Python code"
    }}
  ],
  "answer": "Final answer with the actual computed value"
}}

Rules:
- inputs must use REAL small numbers (e.g. [1.0, 2.0, -1.5], not 'vector u')
- Every step explanation must show actual computed numbers, not just formulas
- Keep LaTeX simple: \\frac{{a}}{{b}}, \\sum, \\sqrt{{x}} etc. No \\begin{{matrix}} blocks
- 3 to 6 steps is ideal
- The answer must be a specific value, not a description

Concept: {concept}"""


@st.cache_data(show_spinner=False)
def generate_with_gemini(concept: str, api_key: str) -> dict:
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(PROMPT.format(concept=concept))

        text = response.text.strip()
        if "```" in text:
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else parts[0]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip().rstrip("`").strip()

        return json.loads(text)

    except json.JSONDecodeError:
        return {"error": "Gemini returned invalid JSON. Try rephrasing the concept name."}
    except Exception as e:
        return {"error": str(e)}
