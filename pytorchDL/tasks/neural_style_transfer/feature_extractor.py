import torch
from torchvision.models import vgg16


class FeatureExtractor(torch.nn.Module):

    def __init__(self, device='cpu'):
        super(FeatureExtractor, self).__init__()

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cuda' if device == 'gpu' else 'cpu')

        self.content_cn_idx = [3]
        self.style_cn_idx = [0, 1, 2, 3, 4]
        vgg = vgg16(pretrained=True).features.to(self.device).eval()
        self.model = vgg[0:11]   # extract up to conv_4 layer

        self.mean_rgb = torch.tensor([0.485, 0.456, 0.406]).to(self.device).view(-1, 1, 1)
        self.std_rgb = torch.tensor([0.229, 0.224, 0.225]).to(self.device).view(-1, 1, 1)

    def forward(self, x):
        cn_idx = 0
        content_features = []
        style_features = []

        # normalize input
        x = (x - self.mean_rgb) / self.std_rgb

        for layer in self.model:
            x = layer(x)
            if isinstance(layer, torch.nn.Conv2d):
                if cn_idx in self.content_cn_idx:
                    content_features.append(x)
                if cn_idx in self.style_cn_idx:
                    style_features.append(x)
                cn_idx += 1

        return content_features, style_features
