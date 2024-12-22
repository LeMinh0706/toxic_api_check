import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

'''This code use pre-trained PhoBERT model and CNN model to detect offensive chat messages.'''

class CNN(nn.Module):
    def __init__(self, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, pad_idx):
        super().__init__()
        
        self.fc_input = nn.Linear(embedding_dim, embedding_dim)
        
        self.conv_0 = nn.Conv1d(in_channels=embedding_dim,
                               out_channels=n_filters,
                               kernel_size=filter_sizes[0])
        
        self.conv_1 = nn.Conv1d(in_channels=embedding_dim,
                               out_channels=n_filters,
                               kernel_size=filter_sizes[1])
        
        self.conv_2 = nn.Conv1d(in_channels=embedding_dim,
                               out_channels=n_filters,
                               kernel_size=filter_sizes[2])
        
        self.conv_3 = nn.Conv1d(in_channels=embedding_dim,
                               out_channels=n_filters,
                               kernel_size=filter_sizes[3])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, encoded):
        embedded = self.fc_input(encoded)
        embedded = embedded.permute(0, 2, 1)
        
        conved_0 = F.relu(self.conv_0(embedded))
        conved_1 = F.relu(self.conv_1(embedded))
        conved_2 = F.relu(self.conv_2(embedded))
        conved_3 = F.relu(self.conv_3(embedded))
        
        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        pooled_3 = F.max_pool1d(conved_3, conved_3.shape[2]).squeeze(2)
        
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2, pooled_3), dim=1))
        
        return self.fc(cat)

def load_models(phobert_path, cnn_path):
    """Load the trained PhoBERT and CNN models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    phobert = torch.load(phobert_path, map_location=device)
    cnn = torch.load(cnn_path, map_location=device)
    
    phobert.eval()
    cnn.eval()
    
    return phobert, cnn

def predict_text(text, phobert, cnn, tokenizer, device):
    """Predict the label for a given text"""
    encoded_dict = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=20,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoded_dict['input_ids'].to(device)
    attention_mask = encoded_dict['attention_mask'].to(device)
    
    with torch.no_grad():
        embedded = phobert(input_ids, attention_mask)[0]
        predictions = cnn(embedded)
        predicted_label = predictions.argmax(dim=1).item()
    
    return predicted_label

def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
    
    try:
        phobert, cnn = load_models(
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
            print(f"\nKết quả: {label_meanings[predicted_label]}")
            
    except Exception as e:
        print(f"Đã xảy ra lỗi: {str(e)}")

if __name__ == "__main__":
    main()