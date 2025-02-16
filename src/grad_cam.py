'''
I HAVE USED THE CODE SIMILAR TO THE EXAMPLE THAT IS PROVIDED IN THE PYTORCH-GRAD-CAM
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def get_gradcam_visualization(model, target_layer, image_tensor, device):
    model.eval()
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=device.type=='cuda')
    input_tensor = image_tensor.unsqueeze(0).to(device)
    grayscale_cam = cam(input_tensor=input_tensor)[0, :]
    
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min())
    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    return visualization

if __name__ == "__main__":
    from model import SimpleCNN
    from data_loader import get_data_loaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    trainloader, testloader = get_data_loaders()
    sample_img, _ = next(iter(testloader))
    sample_img = sample_img[0]
    vis = get_gradcam_visualization(model, model.conv2, sample_img, device)
    plt.imshow(vis)
    plt.title("Grad-CAM Visualization")
    plt.axis('off')
    plt.show()
