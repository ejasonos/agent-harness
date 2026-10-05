from model.ollama_client import OllamaClient


client = OllamaClient()

print("Ollama available:", client.health_check())

print("\nModels:")
for model in client.list_models():
    print(" -", model)

response = client.chat(
    [
        {
            "role": "user",
            "content": "What is 25 + 17? Answer briefly.",
        }
    ]
)

print("\nModel response:")
print(response.content)

print("\nTool calls:")
print(response.tool_calls)