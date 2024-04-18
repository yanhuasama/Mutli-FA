import os
import os.path as osp

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

from .datasets import register

@register('fc100')
class Cifar(Dataset):

    def __init__(self, root_path, split='train', return_path=False, **kwargs):

        if split == 'train':
            THE_PATH = osp.join(root_path, 'train')
            label_list = os.listdir(THE_PATH)
        elif split == 'test':
            THE_PATH = osp.join(root_path, 'test')
            label_list = os.listdir(THE_PATH)
        elif split == 'val':
            THE_PATH = osp.join(root_path, 'val')
            label_list = os.listdir(THE_PATH)
        else:
            raise ValueError('Unkown setname.')

        # for label in label_list:
        #     print(label)
        label_list.sort()
        data = []
        label = []

        folders = [osp.join(THE_PATH, label) for label in label_list if os.path.isdir(osp.join(THE_PATH, label))]

        # Get the images' paths and labels
        for idx, this_folder in enumerate(folders):
            this_folder_images = os.listdir(this_folder)
            this_folder_images.sort()
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(idx)

        self.data = data
        self.label = label
        self.n_classes = len(set(label))
        self.return_path = return_path
        norm_params = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}

        image_size = 84
        normalize = transforms.Normalize(**norm_params)
        self.default_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize,
        ])
        augment = kwargs.get('augment')
        if augment == 'resize':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif augment == 'crop':
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomCrop(image_size, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif augment is None:
            self.transform = self.default_transform
        # Transformation
        # if split == 'train':
        #     image_size = 84
        #     self.transform = transforms.Compose([
        #         transforms.RandomResizedCrop(image_size),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize(**norm_params)])
        #
        # else:
        #     image_size = 84
        #     resize_size = 92
        #
        #     self.transform = transforms.Compose([
        #         transforms.Resize([resize_size, resize_size]),
        #         transforms.CenterCrop(image_size),
        #         transforms.ToTensor(),
        #         transforms.Normalize(**norm_params)])

        def convert_raw(x):
            mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
            std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
            return x * std + mean
        self.convert_raw = convert_raw

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        if self.return_path:
            return image, label, path
        else:
            return image, label



if __name__ == '__main__':
    Cifar('../materials/FC100','test')

