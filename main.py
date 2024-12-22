import torch
from transformers import AutoTokenizer
from cnn import CNN
from model import load_models
from predict import predict_text
from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

import sys
sys.modules['__main__'].CNN = CNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

phobert, cnn = load_models(
    'phobert_cnn_model_part1_task2a_2.pt',
    'phobert_cnn_model_part2_task2a_2.pt'
)
        
app = FastAPI()

class Sentence(BaseModel):
    content: str

@app.post("/check")
def predict(sentence: Sentence):
    label = predict_text(sentence.content, phobert, cnn, tokenizer, device)
    return {"data":label}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
