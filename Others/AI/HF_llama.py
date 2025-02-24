# ECS
# FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04
# RUN apt-get update && apt-get install -y python3-pip
# COPY requirements.txt .
# RUN pip3 install -r requirements.txt
# COPY app.py .
# CMD ["python3", "app.py"]

from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()
model_name = "meta-llama/Llama-3-8b"  # Adjust as needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, 
                                             device_map="auto", 
                                             load_in_4bit=True)

@app.post("/chat")
async def chat(message: str):
    inputs = tokenizer(message, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=50)
    return {"response": tokenizer.decode(outputs[0], skip_special_tokens=True)}