import os
import numpy as np
from torch.utils import data
from torchvision import transforms as T
from skimage import io
import cv2

class ImageFolder(data.Dataset):
    def __init__(self):

        self.root = "./images"
        self.test_list = [os.path.join(self.root, i) for i in os.listdir(self.root)]

    def __getitem__(self, item):
        image_path = self.test_list[item]
        image_id = image_path.split('\\')[-1].split('.')[0]

        image = io.imread(image_path,as_gray=False)
        gradients = cv2.Sobel(image, cv2.CV_64F, 0, 1)
        a = np.uint8(np.maximum(gradients, 0))
        b = np.uint8(np.maximum(-gradients, 0))

        trans = T.Compose([T.ToTensor()])
        image = np.stack([image,a,b],axis=2)
        image = trans(image)

        return image,image_id


    def __len__(self):
        return len(self.test_list)

def get_loader(num_workers=2):
    dataset = ImageFolder()
    loader = data.DataLoader(dataset,batch_size=1,shuffle=False,num_workers=num_workers)
    return loader

