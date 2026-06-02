import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
print("=== KIEM TRA API KEY ===")
print("API Key bat dau bang:", api_key[:15] if api_key else "None")

# 1. Goi thu ListModels
url_list = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
try:
    res = requests.get(url_list)
    if res.status_code == 200:
        print("-> ListModels: OK")
    else:
        print(f"-> ListModels: FAIL (Code {res.status_code})")
except Exception as e:
    print("-> ListModels: ERROR:", e)

# 2. Goi thu generateContent
model = os.getenv("TRAVEL_AGENT_MODEL", "gemini-2.5-flash-lite")
url_gen = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
headers = {"Content-Type": "application/json"}
payload = {"contents": [{"parts": [{"text": "Hello, respond with one word."}]}]}

try:
    res = requests.post(url_gen, json=payload, headers=headers)
    if res.status_code == 200:
        print("-> Call model: OK")
        print("Gemini response:", res.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text"))
    else:
        print(f"-> Call model: FAIL (Code {res.status_code})")
        print("Details:", res.json())
except Exception as e:
    print("-> Call model: ERROR:", e)
