from transformers import (
    XLMRobertaModel,
    AutoTokenizer
)
import torch
import numpy as np
from MultiTaskModel import MultiTaskModel

def load_models(model_path, input_model, device):
    """Load the trained PhoBERT and CNN models"""
    model = MultiTaskModel(input_model = input_model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    return model
def predict_text(text, model, tokenizer, device):
    texts = tokenizer(text,
                    padding='max_length',
                    max_length = 64, truncation=True,
                    return_tensors="pt")
    
    span_preds = []

    input_ids = texts['input_ids'].squeeze(1).to(device)
    attention_mask = texts['attention_mask'].to(device)

    with torch.no_grad():
        span_logits = model(input_ids, attention_mask)
    span_preds.append(span_logits.squeeze().cpu().numpy().flatten())
    span_preds = np.concatenate(span_preds)
    span_preds = (span_preds > 0.5).astype(int)
    return span_preds

def main():
    input_model = XLMRobertaModel.from_pretrained("vinai/phobert-large")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large")
    input_model.resize_token_embeddings(len(tokenizer))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    try:
        model = load_models('model.pt')
        
        label_meanings = {
            0: "Không xúc phạm",
            1: "XÚc phạm người cụ thể",
            2: "Xúc phạm người không cụ thể"
        }
        
        print("\nHệ thống phát hiện nội dung xúc phạm")
        print("Nhập 'quit' để thoát\n")
        
        while True:
            text = input("\nNhập văn bản cần kiểm tra: ")
            if text.lower() == 'quit':
                print('Chào tạm biệt ^_^')
                break
            if not text.strip():
                print("Vui lòng nhập văn bản!")
                continue
            predicted_label = predict_text(text, model, tokenizer, device)
            print(f"\nKết quả: {label_meanings[predicted_label]}")
            
    except Exception as e:
        print(f"Đã xảy ra lỗi: {str(e)}")