import torchvision
import torchvision.transforms as transforms
import torch

def get_data_loaders(batch_size=128, num_workers=2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

  
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=num_workers)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,shuffle=False, num_workers=num_workers)
    return trainloader, testloader

if __name__ == "__main__":
    trainloader, testloader = get_data_loaders()
    print("dATA LOADER IS NOW CREATED")
