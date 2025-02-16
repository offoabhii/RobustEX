
import torchattacks
import torch

def get_fgsm_attack(model, eps=0.03):
    return torchattacks.FGSM(model, eps=eps)

def generate_adversarial_example(model, image, label, attack):
    image = image.unsqueeze(0).to(next(model.parameters()).device)
    label = torch.tensor([label]).to(next(model.parameters()).device)
    adv_image = attack(image, label)
    return adv_image.squeeze(0).detach().cpu()

if __name__ == "__main__":
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    sample_img, sample_label = testset[0]
    from model import SimpleCNN
    model = SimpleCNN()
    attack = get_fgsm_attack(model)
    adv_img = generate_adversarial_example(model, sample_img, sample_label, attack)
    print("Adversarial example generated!")

