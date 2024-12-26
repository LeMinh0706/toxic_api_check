import torch
from transformers import AutoTokenizer
from cnn import CNN
from model import load_models
from predict import predict_text
from typing import Union
import uvicorn
from sensitive import sensitive
from ultralytics import YOLOv10
from datetime import datetime
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil

import sys

yolo = YOLOv10('best.pt')

sys.modules['__main__'].CNN = CNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

phobert, cnn = load_models(
    'phobert_cnn_model_part1_task2a_2.pt',
    'phobert_cnn_model_part2_task2a_2.pt',
    device
)
        
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):

    try:
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        file_extension = file.filename.split('.')[-1] 
        new_filename = f"{current_time}.{file_extension}"

        file_location = f"uploads/{new_filename}"

        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)

        label, message = sensitive(yolo, file_location)

        return {"code":200,"message":message, "data":{"file":file_location, "label":label}}

    except Exception as e:
        raise HTTPException(status_code=200, detail={"code": 500, "message": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9200)
