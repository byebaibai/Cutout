import glob

import cv2
import numpy as np
import torch

MEAN = [0.4560, 0.4472, 0.4155]

class FastDataset(object):
    def __init__(self, ImageDir, LabelDir):
        self.img_dir = ImageDir
        self.map_dir = LabelDir
        self.img_list = glob.glob(ImageDir + '*' + '.jpg')
        self.label_list = []

        for img_path in self.img_list:
            img_name = img_path.split("/")[-1].split(".")[0]
            self.label_list.append(LabelDir + img_name + '.png')

        print("train images: ", len(self.img_list))
        print("train labels: ", len(self.label_list))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        flip_choice = np.random.randint(1, 100)
        image = cv2.imread(self.img_list[idx])
        label = cv2.imread(self.label_list[idx])
        label = np.array(label != 0, dtype=np.float) * 255.0

        if flip_choice % 2 == 0:
            image = cv2.flip(image, 1)
            label = cv2.flip(image, 1)

        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]

        image = image[:, :, ::-1]
        label = label[:, :, ::-1]

        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (256, 256), interpolation=cv2.INTER_LINEAR)

        image = (image.astype(np.float32) / 255.0 - MEAN)

        image = image.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)
        label = label[0:1, :, :]

        sample = {'image': torch.from_numpy(image), 'label': torch.from_numpy(label / 255.0)}
        return sample

