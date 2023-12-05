import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

def setup(rank, world_size):
    # Initialize the distributed backend
    dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=rank, world_size=world_size)

def cleanup():
    # Clean up the distributed backend
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    torch.manual_seed(42)  # Set seed for reproducibility

    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Set up DistributedSampler
    train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, sampler=train_sampler)

    # Define a simple CNN model
    class SimpleCNN(nn.Module):
        class SimpleCNN(nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
                self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                self.fc1 = nn.Linear(32 * 8 * 8, 128)
                self.fc2 = nn.Linear(128, 10)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = self.pool(F.relu(self.conv2(x)))
                x = x.view(-1, 32 * 8 * 8)
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x

    model = SimpleCNN()
    model = DDP(model, device_ids=[rank])

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(5):  # Adjust the number of epochs as needed
        model.train()
        train_sampler.set_epoch(epoch)
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    cleanup()

if __name__ == "__main__":
    world_size = 4  # Total number of processes (GPUs/Nodes)
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
