from torchvision import transforms
from torchvision import datasets
import torch
import numpy as np

# Change it for your dataset directory
DATASET_PATH = '/disk3/noga/dataset/'



class IndexedNoisyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, noise_ratio=0.4, noise_type='uniform', num_cls = 10, dataset_name='cifar10'):
        np.random.seed(67012)
        if noise_type == 'uniform':
            self.noisy_n = int(len(dataset) * noise_ratio)
            self.noisy_i = np.random.choice(len(dataset), self.noisy_n, replace=False)
            self.noisy_labels = torch.zeros((len(dataset),), dtype=torch.int)
            self.noisy_labels[self.noisy_i] = torch.randint(0, num_cls, (self.noisy_n,), dtype=torch.int)
        else:
            raise NotImplementedError
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        if index in self.noisy_i:
            target = self.noisy_labels[index].item()
        return data, target, index

    def __len__(self):
        return len(self.dataset)


def cifar10_dataset(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    cifar_dataset = datasets.CIFAR10(DATASET_PATH, train=True, transform=transform_train, target_transform=None,
                                                 download=True)
    cifar_test_dataset = datasets.CIFAR10(DATASET_PATH, train=False, transform=transform_test,
                                                      target_transform=None, download=True)
    return cifar_dataset, cifar_test_dataset

def cifar100_dataset(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),  (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),  (0.2675, 0.2565, 0.2761)),
    ])

    cifar_dataset = datasets.CIFAR100(DATASET_PATH, train=True, transform=transform_train, target_transform=None,
                                                 download=True)
    cifar_test_dataset = datasets.CIFAR100(DATASET_PATH, train=False, transform=transform_test,
                                                      target_transform=None, download=True)
    return cifar_dataset, cifar_test_dataset


def indexed_cifar(batch_size=128, noise_ratio=0.4, noise_type='uniform', dataset_name='cifar10', num_cls=10):
    if dataset_name == 'cifar10':
        train_dataset_clean, test_dataset_clean = cifar10_dataset(batch_size)
    else:
        train_dataset_clean, test_dataset_clean = cifar100_dataset(batch_size)

    train_dataset_noisy = IndexedNoisyDataset(train_dataset_clean, noise_ratio, noise_type, dataset_name=dataset_name, num_cls=num_cls)
    train_data_loader_noisy = torch.utils.data.DataLoader(train_dataset_noisy, batch_size=batch_size, shuffle=True, num_workers=0)
    train_data_loader_noisy_fw = torch.utils.data.DataLoader(train_dataset_noisy, batch_size=batch_size, shuffle=True, num_workers=0)

    test_data_loader = torch.utils.data.DataLoader(test_dataset_clean, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_data_loader_noisy, test_data_loader, train_data_loader_noisy_fw


