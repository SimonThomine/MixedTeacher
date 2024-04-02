import timm
import torch.nn as nn

class studentEffNet(nn.Module):
    def __init__(
        self,
        backbone_name="efficientnet_b0",
        out_indices=[3,4]
    ):
        super(studentEffNet, self).__init__()     
        self.feature_extractor = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
            out_indices=out_indices 
        )
        self.modelName = backbone_name

        
    def forward(self, x):
        features_t = self.feature_extractor(x)
        return features_t