import torch
from transformers import AutoTokenizer
from cnn import CNN 
from model import load_models
from predict import predict_text
def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
    
    try:
        phobert, c nn = load_models(
            'phobert_cnn_model_part1_task2a_2.pt',
            'phobert_cnn_model_part2_task2a_2.pt'
        )
        
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
            predicted_label = predict_text(text, phobert, cnn, tokenizer, device)
            print(f"\nKết quả: {predicted_label}")
            
    except Exception as e:
        print(f"Đã xảy ra lỗi: {str(e)}")

if __name__ == "__main__":
    main()