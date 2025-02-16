import torch
import torch.optim as optim
from model import SimpleCNN
from data_loader import get_data_loaders
import torchattacks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, trainloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[Epoch {epoch+1}, Batch {i+1}] loss: {running_loss/100:.3f}")
                running_loss = 0.0
    print("Finished Clean Training")

def train_model_adv(model, trainloader, criterion, optimizer, attack, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            adv_inputs = attack(inputs, labels)      # USED TO GENERATE ADVERSIAL ATTACK
            
            mixed_inputs = torch.cat([inputs, adv_inputs], dim=0)
            mixed_labels = torch.cat([labels, labels], dim=0)
            
            optimizer.zero_grad()
            outputs = model(mixed_inputs)
            loss = criterion(outputs, mixed_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[Adv Training Epoch {epoch+1}, Batch {i+1}] loss: {running_loss/100:.3f}")
                running_loss = 0.0
    print("Finished Adversarial Training")

if __name__ == "__main__":
    trainloader, _ = get_data_loaders()
    model = SimpleCNN().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Clean training
    print("Starting Clean Training")
    train_model(model, trainloader, criterion, optimizer, num_epochs=10)
    
    # Save the clean model if desired:
    torch.save(model.state_dict(), "baseline_model.pth")
    
    # Set up the attack from torchattacks
    attack = torchattacks.FGSM(model, eps=0.03)
    print("Starting Adversarial Training")
    train_model_adv(model, trainloader, criterion, optimizer, attack, num_epochs=5)
    
    torch.save(model.state_dict(), "adv_trained_model.pth")
