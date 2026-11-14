# Created: 14/Nov/2026

from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = os.getenv("NVIDIA_API_KEY")
)

completion = client.chat.completions.create(
  model=os.getenv("NVIDIA_MODEL"),
  messages=[{"role":"user","content":"Which number is larger, 9.11 or 9.8?"}],
  temperature=1,
  top_p=0.95,
  max_tokens=8192,
  stream=False
)

print(completion.choices[0].message.content)