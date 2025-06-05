# c:\Users\brade\Desktop\Vibecoding4Profit\LLM-WEB-APP\test_ollama.py
import ollama
import os

print(f"Attempting to connect to Ollama...")
print(f"Python executable: {os.sys.executable}")
print(f"OLLAMA_HOST environment variable: {os.environ.get('OLLAMA_HOST')}")

try:
    list_response = ollama.list() # Changed variable name for clarity
    print("\nSuccessfully connected to Ollama.")
    print(f"Raw models response: {list_response}") 
    print("Available models:")
    # Correctly check for the 'models' attribute and iterate through ModelResponse objects
    if list_response and hasattr(list_response, 'models') and list_response.models:
        for model_item in list_response.models: # model_item is a ModelResponse object
            print(f"- {model_item.name} (Size: {model_item.size/(1024**3):.2f}GB, Modified: {model_item.modified_at})")
    else:
        print("No models found, or the response format was unexpected (expected an object with a 'models' attribute list).")
except Exception as e:
    print(f"\nError accessing Ollama from Python: {e}")
    print("Please ensure the Ollama service is running and accessible.")
    print("If Ollama is running on a custom host/port, ensure the OLLAMA_HOST environment variable is set accordingly (e.g., http://mycustomhost:12345).")
