import torch

def load_models(phobert_path, cnn_path, device):
    
    phobert = torch.load(phobert_path, map_location=device)
    cnn = torch.load(cnn_path, map_location=device)
    
    phobert.eval()
    cnn.eval()
    
    return phobert, cnn