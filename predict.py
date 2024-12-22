import torch

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

