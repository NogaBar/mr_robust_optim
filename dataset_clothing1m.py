import pandas as pd
from torchvision.datasets.folder import default_loader
import torch
from torchvision import transforms
from pickle import dump, load
import os

IMAGE_SIZE = 256
CROP_SIZE = 224

class Clothing1M(torch.utils.data.Dataset):
    def get_images_labels(self, split):
        all_noisy = pd.read_csv('clothing1m/noisy_label_kv.txt', names=['img_id', 'label'], delimiter=' ')
        all_clean = pd.read_csv('clothing1m/clean_label_kv.txt', names=['img_id', 'label'], delimiter=' ')

        if split == 'train':
            split_img_ids = pd.read_csv('clothing1m/noisy_train_key_list.txt', delimiter=' ', names=['img_id'])
            imgs_labels = all_noisy[all_noisy['img_id'].isin(split_img_ids['img_id'])]
        else:
            split_img_ids = pd.read_csv(f'clothing1m/clean_{split}_key_list.txt', names=['img_id'], delimiter=' ')
            imgs_labels = all_clean[all_clean['img_id'].isin(split_img_ids['img_id'])]
        return imgs_labels


    def __init__(self, split):
        self.imgs_labels = self.get_images_labels(split)
        self.split = split

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize([IMAGE_SIZE, IMAGE_SIZE]),
                transforms.RandomCrop(IMAGE_SIZE, padding=8),
                transforms.CenterCrop(CROP_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize([IMAGE_SIZE, IMAGE_SIZE]),
                transforms.CenterCrop(CROP_SIZE),
                transforms.ToTensor(),
                transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
            ])

    def __getitem__(self, index):
        data, target = self.imgs_labels.iloc[index]
        data = self.transform(default_loader(os.path.join('clothing1m', data)))
        if self.split == 'test' or self.split == 'val':
            return data, target
        return data, target, index


    def __len__(self):
        return len(self.imgs_labels)

def main():
    train = Clothing1M('train')
    print(train[999999])
    print(len(train))

    loader = torch.utils.data.DataLoader(train, batch_size=10, num_workers=1, shuffle=True)

    for data in loader:
        input, target, i = data
        print('input', input.shape)
        print('target', target)
        print('i', i)
        break








if __name__ == '__main__':
    main()