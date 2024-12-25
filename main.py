import torch
from transformers import AutoTokenizer
from cnn import CNN
from model import load_models
from predict import predict_text
from typing import Union
import uvicorn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import sys
sys.modules['__main__'].CNN = CNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

phobert, cnn = load_models(
    'phobert_cnn_model_part1_task2a_2.pt',
    'phobert_cnn_model_part2_task2a_2.pt',
    device
)
        
app = FastAPI()

class Sentence(BaseModel):
    content: str

@app.post("/check")
def predict(sentence: Sentence):
    try:
        label = predict_text(sentence.content, phobert, cnn, tokenizer, device)
        meanings = {
            0: "",
            1: "Có nội dung phản cảm, vui lòng chỉnh sửa",
            2: "Có từ ngữ mang tính xúc phạm"
        }
        return {"code":200,"message":meanings[label], "data":{"label":label}}
    except Exception as e:
        raise HTTPException(status_code=200, detail={"code": 500, "message": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9200)
