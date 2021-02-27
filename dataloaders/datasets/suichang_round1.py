from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr

class SuichangSegmentation(Dataset):
    """
    Suichang round1 dataset
    """
    NUM_CLASSES = 10

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('suichang_round1'),
                 split='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self.split = split
        self._base_dir = base_dir
        assert split in ['train', 'val', 'test']
        if split == 'train':
            self._image_dir = os.path.join(self._base_dir, 'suichang_round1_train_210120', 'original_images')
            self._seg_dir = os.path.join(self._base_dir, 'suichang_round1_train_210120', 'segmentation_minus_one')
        elif split == 'test':
            self._image_dir = os.path.join(self._base_dir, 'suichang_round1_test_partA_210120')
            self._seg_dir = None
        self.args = args

        self.images = []
        self.segs = []

        for img_name in os.listdir(self._image_dir):
            self.images.append(os.path.join(self._image_dir, img_name))
            if split == 'train':
                self.segs.append(os.path.join(self._seg_dir, img_name.replace('.tif', '.png')))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.split == 'train':
            _img, _target = self._make_img_gt_point_pair(index)
            sample = {'image': _img, 'label': _target}
            return self.transform_tr(sample)
        elif self.split == 'test':
            return self.transform_test(Image.open(self.images[index]))


    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index])
        _target = Image.open(self.segs[index])

        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.Normalize((0.190, 0.235, 0.222, 0.527), (0.127, 0.122, 0.119, 0.199)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.Normalize((0.190, 0.235, 0.222, 0.527), (0.127, 0.122, 0.119, 0.199)),
            tr.ToTensor()])

    def transform_test(self, image):
        composed_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.190, 0.235, 0.222, 0.527), (0.127, 0.122, 0.119, 0.199))
            ])
        return composed_transforms(image)

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    voc_train = SuichangSegmentation(args, split='train')

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in tqdm(enumerate(dataloader)):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='pascal')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            # break
            continue

    plt.show(block=True)


