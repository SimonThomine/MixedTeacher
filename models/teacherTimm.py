import timm
import torch.nn as nn

class teacherTimm(nn.Module):
    def __init__(
        self,
        backbone_name="resnet18",
        out_indices=[1, 2, 3]
    ):
        super(teacherTimm, self).__init__()     
        self.feature_extractor = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
            out_indices=out_indices 
        )
        self.modelName = backbone_name
        self.feature_extractor.eval() 
        for param in self.feature_extractor.parameters():
            param.requires_grad = False   
        self.avgPool=nn.AvgPool2d((2,2))
        
    def forward(self, x):
        features_t = self.feature_extractor(x)
        if (self.modelName=="resnet18"):
            features_t = [self.avgPool(f) for f in features_t]
        return features_t
