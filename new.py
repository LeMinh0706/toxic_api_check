import torch
import torch.nn as nn
from transformers import (
    XLMRobertaModel,
    AutoTokenizer
)
from MultiTaskModel import MultiTaskModel
import torch
import numpy as np



def predict_text(text, model, tokenizer, device):
    texts = tokenizer(text,
                    padding='max_length',
                    max_length = 20, truncation=True,
                    return_tensors="pt", add_special_tokens=True, return_attention_mask=True,)
    
    model.eval()

    span_preds = []

    input_ids = texts['input_ids'].squeeze(1).to(device)
    attention_mask = texts['attention_mask'].to(device)

    with torch.no_grad():
        span_logits = model(input_ids, attention_mask)

    span_preds.append(span_logits.squeeze().cpu().numpy().flatten())
    span_preds = np.concatenate(span_preds)
    span_preds = (span_preds > 0.5).astype(int)
    # span_preds = span_logits.argmax(dim=1).item()
    return span_preds


def main():
    input_model = XLMRobertaModel.from_pretrained("vinai/phobert-large")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large")
    input_model.resize_token_embeddings(len(tokenizer))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        model = MultiTaskModel(input_model = input_model)
        # model.load_state_dict(torch.load("model.pt", map_location=device))
        
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
            print(f"\nNhãn: {predicted_label}")
            # print(f"\nKết quả: {np.max(predicted_label)}")
            
    except Exception as e:
        print(f"Đã xảy ra lỗi: {str(e)}")

if __name__ == "__main__":
    main()