import torchvision
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np


class MSCTD(Dataset):
    def __init__(self, root, split, image_transform=None, text_transform=None, sentiment_transform=None,
                 has_image=True, has_text=True, text_path=None, image_path=None, sentiment_path=None,
                 image_index_path=None):
        """
        :param root: root path of the dataset
        :param split: train, dev, test
        :param image_transform: transform for image
        :param text_transform: transform for text
        :param sentiment_transform: transform for sentiment
        :param has_image: if the dataset has image
        :param has_text: if the dataset has text
        :param text_path: path of the text file
        :param image_path: path of the image folder
        :param sentiment_path: path of the sentiment file
        :param image_index_path: path of the image index file

        :return: combination of image, sentiment, text, image_index

        Example:
        >>> from torchvision import transforms
        >>> image_transform = transforms.Compose([
        >>>     transforms.ToTensor(),
        >>>     transforms.Resize((640, 1280)),
        >>>     transforms.Lambda(lambda x: x.permute(1, 2, 0))
        >>> ])
        >>> text_transform = None
        >>> sentiment_transform = None
        >>> dataset = MSCTD(root='data', split='train', image_transform=image_transform,
        >>>                 text_transform=text_transform, sentiment_transform=sentiment_transform)
        >>> image, text, sentiment = dataset[0]

        """
        self.root = root
        self.split = split
        if has_image:
            self.image = torchvision.datasets.ImageFolder(self.root, transform=image_transform)
            self.image_transform = image_transform
            if image_path is None:
                self.image_path = os.path.join(root, split, 'image')
            else:
                self.image_path = os.path.join(root, image_path)
        if has_text:
            self.text = []
            self.text_transform = text_transform
            if text_path is None:
                self.text_path = os.path.join(root, split, 'english_' + split + '.txt')
            else:
                self.text_path = os.path.join(root, text_path)
        if sentiment_path is None:
            self.sentiment_path = os.path.join(root, split, 'sentiment_' + split + '.txt')
        else:
            self.sentiment_path = os.path.join(root, sentiment_path)
        if image_index_path is None:
            self.image_index_path = os.path.join(root, split, 'image_index_' + split + '.txt')
        else:
            self.image_index_path = os.path.join(root, image_index_path)
        self.sentiment = []
        self.image_index = []
        self.sentiment_transform = sentiment_transform
        self.load_data()
        
    def load_data(self):
        self.sentiment = np.loadtxt(self.sentiment_path, dtype=np.uint8)
        if hasattr(self, 'text'):
            with open(self.text_path, 'r') as f:
                self.text = f.readlines()
        with open(self.image_index_path, 'r') as f:
            for line in f:
                index = line.strip()[1:-1].split(', ')
                index = [int(i) for i in index]
                self.image_index.append(index)
                    
    def __getitem__(self, index):
        if hasattr(self, 'image'):
            image = self.image[index][0]
        if hasattr(self, 'text'):
            text = self.text[index]
            if self.text_transform:
                text = self.text_transform(text)
        sentiment = self.sentiment[index]
        # image_index = self.image_index[index]
        if self.sentiment_transform:
            sentiment = self.sentiment_transform(sentiment)
        # if the class has image and text attribute return image, text, sentiment
        if hasattr(self, 'image') and hasattr(self, 'text'):
            return image, text, sentiment
        # if the class has image attribute return image, sentiment
        elif hasattr(self, 'image'):
            return image, sentiment
        # if the class has text attribute return text, sentiment
        elif hasattr(self, 'text'):
            return text, sentiment
        else:
            raise Exception('Either has_image or has_text should be True')


    def __len__(self):
        return len(self.sentiment)
    