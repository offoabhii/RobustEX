
import torch
from src.data_loader import get_data_loaders
from src.model import SimpleCNN
from src.train import train_model, train_model_adv
from src.attacks import get_fgsm_attack, generate_adversarial_example
from src.grad_cam import get_gradcam_visualization
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    trainloader, testloader = get_data_loaders()
    

    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
        print("Starting Clean Training")
    train_model(model, trainloader, criterion, optimizer, num_epochs=10)
    

    torch.save(model.state_dict(), "baseline_model.pth")
    attack = get_fgsm_attack(model, eps=0.03)
    for images, labels in testloader:
        sample_img = images[0]
        sample_label = labels[0]
        adv_img = generate_adversarial_example(model, sample_img, sample_label, attack)
        break
    def imshow(img, title=""):
        img = img / 2 + 0.5  # unnormalize if normalized with mean 0.5
        npimg = img.numpy().transpose(1, 2, 0)
        plt.imshow(np.clip(npimg, 0, 1))
        plt.title(title)
        plt.axis('off')
        plt.show()
    
    imshow(sample_img.cpu(), "Clean Image")
    imshow(adv_img, "Adversarial Image")
    print("Starting Adversarial Training")
    train_model_adv(model, trainloader, criterion, optimizer, attack, num_epochs=5)
    torch.save(model.state_dict(), "adv_trained_model.pth")
    vis_before = get_gradcam_visualization(model, model.conv2, sample_img, device)
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(vis_before)
    plt.title("Grad-CAM After Adv Training")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
