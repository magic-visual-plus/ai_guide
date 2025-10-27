import torch
from . import model_utils
from .models import pt

class ForegroundExtractor:
    def __init__(self, model_path, volume_size=4, threshold=0.5):
        self.model = pt.PointTransformerPointcept(input_size=6)
        self.model.load_state_dict(torch.load(model_path))
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self.volume_size = volume_size
        self.threshold = threshold
    
    def extract(self, pcd, threshold=0.5):
        pred_index, _ = model_utils.predict_pcd_pt(pcd, self.model, self.device, self.threshold, self.volume_size)
        
        return pcd.select_by_index(pred_index)
    pass