import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from mpmath import mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as transforms
import os


def setup(rank, world_size, backend='nccl'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12345'

    # Initialize the distributed backend
    dist.init_process_group(backend=backend, init_method='env://', rank=rank, world_size=world_size)


def cleanup():
    # Clean up the distributed backend
    dist.destroy_process_group()


# Function to set up the model for DDP
def setup_model(rank, world_size):
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

        # Initialize model and wrap with DDP
    model = SimpleCNN()
    model = DDP(model, device_ids=[rank])

    return model, train_loader


# Function for training loop
def train(rank, world_size):
    setup(rank, world_size)

    model, train_loader = setup_model(rank, world_size)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(5):  # Adjust the number of epochs as needed
        model.train()
        train_loader.sampler.set_epoch(epoch)
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    cleanup()


# Define main function for running distributed training
if __name__ == "__main__":
    world_size = 4  # Total number of processes (GPUs/Nodes)
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
