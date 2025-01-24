import torch
from torchvision import transforms
from multi_mnist_loader import MNIST

# Setup Augmentations


def global_transformer():
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])


def get_datasetmnist(params, configs):
    
    train_dst = MNIST(root=configs['mnist']['path'], train=True, download=True, transform=global_transformer(), multi=True)
    train_loader = torch.utils.data.DataLoader(train_dst, batch_size=64, shuffle=True, num_workers=4)

    val_dst = MNIST(root=configs['mnist']['path'], train=False, download=True, transform=global_transformer(), multi=True)
    val_loader = torch.utils.data.DataLoader(val_dst, batch_size=64, shuffle=True, num_workers=4)
    return train_dst, train_loader, val_dst, val_loader 