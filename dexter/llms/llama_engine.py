from dotenv import load_dotenv
load_dotenv()
import os
from transformers import AutoTokenizer
from huggingface_hub import InferenceClient
from groq import Groq

class LlamaEngine:

    def __init__(self, data, model_name="meta-llama/Llama-2-7b-chat-hf", temperature=0.3, top_n=1, max_new_tokens=256):

        #access_token_read = os.environ.get("huggingface_token")
        #if not access_token_read:
             #raise ValueError("huggingface_token environment variable is required")

        self.model_name = model_name
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        #self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token_read)
        
        self.temperature = temperature
        self.data = data
        self.top_n = top_n
        self.max_new_tokens=max_new_tokens

    def get_llama_completion(self, system_prompt: str, user_prompt: str):
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            top_p=0.95,
            max_tokens=self.max_new_tokens,
            seed=42  # Fixed seed for reproducibility
        )
    
        return response.choices[0].message.content