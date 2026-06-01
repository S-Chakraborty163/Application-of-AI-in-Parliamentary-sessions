import os, json, re, time
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(override=True)
cf_account = os.environ.get('CLOUDFLARE_ACCOUNT_ID')
cf_token = os.environ.get('CLOUDFLARE_API_TOKEN')
client = OpenAI(base_url=f'https://api.cloudflare.com/client/v4/accounts/{cf_account}/ai/v1', api_key=cf_token, max_retries=0)
prompt = '''You are an impartial Judge evaluating an answer.
Context:
test

Task/Question:
test

Answer A:
test

Score Answer A based strictly on this mathematical rubric:
1. Faithfulness (0-100): Start at 100. Deduct points heavily if the answer uses information outside the provided Context.
2. Relevance (0-100): Start at 100. Deduct points heavily if it fails to answer the core Task/Question.
3. Formatting (0 or 100): 100 if it followed all structural/formatting constraints requested in the Task/Question, else 0. If no constraint was requested, give 100.

Return ONLY a valid JSON object matching this schema, nothing else:
{"faithfulness": 90, "relevance": 80, "formatting": 100}'''
try:
    msg = [{'role': 'user', 'content': prompt}]
    res = client.chat.completions.create(model='@cf/meta/llama-3.3-70b-instruct-fp8-fast', messages=msg, temperature=0.0)
    print('Raw response:', repr(res.choices[0].message.content))
except Exception as e:
    print('Error:', e)
