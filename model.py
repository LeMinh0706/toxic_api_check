import torch

def load_models(phobert_path, cnn_path):
    """Load the trained PhoBERT and CNN models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    phobert = torch.load(phobert_path, map_location=device)
    cnn = torch.load(cnn_path, map_location=device)
    
    phobert.eval()
    cnn.eval()
    
    return phobert, cnn