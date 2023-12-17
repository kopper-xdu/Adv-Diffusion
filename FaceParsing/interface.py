from .networks import get_model
import torch
import torch.nn.functional as F


class FaceParsing(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.model = get_model('FaceParseNet50', pretrained=False)
        self.model.load_state_dict(torch.load('pretrained_model/38_G.pth', map_location='cpu'))
        self.model.eval()
        
    def forward(self, x):
        # (B, 3, 512, 512)
        outputs = self.model(x)[0][-1]
        imsize = x.shape[2]
        inputs = F.interpolate(input=outputs, size=(imsize, imsize), mode='bilinear', align_corners=True)

        pred_batch = torch.argmax(inputs, dim=1)
        
        return pred_batch
    
    
if __name__ == '__main__':
    from PIL import Image
    from torchvision import transforms
    import cv2
    
    input = Image.open('./17082.png')
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    input = transform(input).unsqueeze(0).cuda()
    
    model = FaceParsing()
    out = model(input)
    # print(out)
    
    cv2.imwrite('2.png', out[0].cpu().numpy())
    