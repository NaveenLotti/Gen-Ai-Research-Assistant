import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ GEMINI_API_KEY not found in .env")
    exit()

print("=" * 60)
print("GEMINI API TEST")
print("=" * 60)

print("\nAPI key found:", api_key[:8] + "..." + api_key[-4:])

try:
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents="Reply with exactly: Gemini API is working."
    )

    print("\n✅ API REQUEST SUCCESSFUL")
    print("\nGemini response:")
    print(response.text)

except Exception as e:

    print("\n❌ API REQUEST FAILED")

    print("\nError type:")
    print(type(e).__name__)

    print("\nError:")
    print(e)