import torch
import torch.nn as nn
import torchvision.models as models

class MelanomaClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(MelanomaClassifier, self).__init__()
        
        self.model = models.efficientnet_b4(pretrained=True)
        self.model._fc = nn.Linear(in_features=1792, out_features=num_classes, bias=True)
        
    def forward(self, x):
        x = self.model(x)
        return x
