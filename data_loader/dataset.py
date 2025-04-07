import os
import pathlib
import json
import random
import numpy as np
import torchvision as tv
import copy
import torch

from PIL import Image
from torch import distributed
from utils import transformmasks
from utils import transformsgpu
from data_loader import DATASETS_IMG_DIRS
from data_loader import custom_transforms as tr
from base.base_dataset import BaseDataset, lbl_contains_any, lbl_contains_all

#Target masking
class VOCSegmentationIncremental(BaseDataset):
    """
    PascalVoc dataset
    """
    def __init__(
        self,
        test=False, val=False, setting='overlap', classes_idx_new=[], classes_idx_old=[],
        transform=True, transform_args={}, masking_value=0, idxs_path=None,
    ):
        if setting not in ['sequential', 'disjoint', 'overlap']:
            raise ValueError('Wrong setting entered! Please use one of sequential, disjoint, overlap')

        super().__init__(
            transform_args=transform_args,
            base_dir=pathlib.Path(DATASETS_IMG_DIRS['voc']),
            transform=transform,
        )
        self.setting = setting  #'overlap'
        self.classes_idx_old = classes_idx_old 
        self.classes_idx_new = classes_idx_new 

        self.test = test    #False
        self.val = val      #False
        self.train = not (self.test or self.val)    #True

        self.masking_value = masking_value  #0

        if self.train:
            self.split = 'train_aug'
        else:
            self.split = 'val'

        #dataset lujing
        if 'aug' not in self.split:
            self._image_dir = self._base_dir / "JPEGImages"
            self._cat_dir = self._base_dir / "SegmentationClass"
        else:
            self._image_dir = self._base_dir
            self._cat_dir = self._base_dir
        _splits_dir = self._base_dir / "ImageSets" / "Segmentation"

        self.im_ids = []
        self.categories = []
        #_image: image      _cat:       self.images.append(_image)   self.categories.append(_cat)
        if (idxs_path is not None) and (os.path.exists(idxs_path)):
            self.im_ids = np.load(idxs_path, allow_pickle=True).tolist()   
            for x in self.im_ids:
                if 'aug' not in self.split:
                    _image = self._image_dir / f"{x}.jpg"
                    _cat = self._cat_dir / f"{x}.png"
                else:
                    _image = self._image_dir / x.split()[0][1:]
                    _cat = self._cat_dir / x.split()[1][1:]

                assert _image.is_file(), _image
                assert _image.is_file(), _cat
                
                self.images.append(_image)      
                self.categories.append(_cat)   
        else:
            if distributed.get_rank() == 0:
                print("Filtering images....")

            lines = (_splits_dir / f"{self.split}.txt").read_text().splitlines()
            for ii, line in enumerate(lines):
                if (ii % 1000 == 0) and (distributed.get_rank() == 0):
                    print(f"[{ii} / {len(lines)}]")
                    
                if 'aug' not in self.split:
                    _image = self._image_dir / f"{line}.jpg"
                    _cat = self._cat_dir / f"{line}.png"
                else:
                    _image = self._image_dir / line.split()[0][1:]
                    _cat = self._cat_dir / line.split()[1][1:]
                assert _image.is_file(), _image
                assert _cat.is_file(), _cat

                cat = Image.open(_cat)
                cat = np.array(cat, dtype=np.uint8)

                if (self.train or self.val):
                    # Remove the sample if the g.t mask does not contain new class
                    if not lbl_contains_any(cat, self.classes_idx_new):
                        continue
                    # Unique set
                    # : Remove the sample if the g.t mask contains any other labels that not appeared yet.
                    if (self.train) and (self.setting == 'disjoint' or self.setting == 'sequential'):
                        if not lbl_contains_all(cat, list(set(self.classes_idx_old + self.classes_idx_new + [0, 255]))):
                            continue
                else:  # Test
                    if not lbl_contains_any(cat, list(set(self.classes_idx_old + self.classes_idx_new))):
                        continue

                self.im_ids.append(line)        #list: ['/JPEGImages/2007_000032.jpg, /SegmentationClassAug/2007_000032.png']
                self.images.append(_image)     
                self.categories.append(_cat)    

            if (idxs_path is not None) and (distributed.get_rank() == 0):
                np.save(idxs_path, np.array(self.im_ids))

        assert len(self.images) == len(self.categories)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)    

        sample = {"image": _img, "label": _target}

        if self.transform:
            if self.split in ["trainval_aug", "trainval", "train_aug", "train"]:
                sample['image'], sample['label'] = self.transform_tr(sample)
            elif self.split in ["val_aug", "val"]:
                sample['image'], sample['label'] = self.transform_val(sample)
        else:
            sample['image'], sample['label'] = self.transform_test(sample)

 

        # Target masking  
        sample['label'] = self.transform_target_masking(sample['label'])  
        sample["image_name"] = str(self.im_ids[index])        
        
        
        return sample   #image image_name label

    def _make_img_gt_point_pair(self, index):    
        _img = Image.open(self.images[index]).convert("RGB")
        _target = Image.open(self.categories[index])
        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = tr.Compose(
            [
                tr.RandomResizedCrop(
                    self.transform_args['crop_size'],
                    (0.5, 2.0)
                ),
                tr.RandomHorizontalFlip(),
                tr.ToTensor(),
                tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
               
            ]
        )
        return composed_transforms(sample['image'], sample['label'])

    def transform_val(self, sample):
        composed_transforms = tr.Compose(
            [
                tr.Resize(size=self.transform_args['crop_size']),
                tr.CenterCrop(self.transform_args['crop_size']),
                tr.ToTensor(),
                tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return composed_transforms(sample['image'], sample['label'])

    def transform_test(self, sample):
        composed_transforms = tr.Compose(
            [
                tr.ToTensor(),
                tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return composed_transforms(sample['image'], sample['label'])

    def transform_target_masking(self, target):
        if self.test:
            # Masking future class object
            # MiB: 255 / PLOP: 0
            label_list = list(set(self.classes_idx_old + self.classes_idx_new + [0, 255]))
            target_transform = tv.transforms.Lambda(
                lambda t: t.apply_(lambda x: x if x in label_list else self.masking_value)
            )
            return target_transform(target)

        else:  
            if self.masking_value is None:
                return target
            if self.setting in ['disjoint', 'overlap']:
                label_list = list(set(self.classes_idx_new + [0, 255]))     
                target_transform = tv.transforms.Lambda(lambda t: t.apply_(lambda x: x if x in label_list else self.masking_value))
                return target_transform(target)

    def transform_target_truebackground_masking(self, target):
        if self.test:

            label_list = list(set(self.classes_idx_old + self.classes_idx_new + [0, 255]))
            target_transform = tv.transforms.Lambda(
                lambda t: t.apply_(lambda x: x if x in label_list else self.masking_value)
            )
            return target_transform(target)
        else:  
            if self.masking_value is None:
                return target
            if self.setting in ['disjoint', 'overlap']:
                label_list = list(set(self.classes_idx_old + self.classes_idx_new + [0, 255]))
                target_transform = tv.transforms.Lambda(lambda t: t.apply_(lambda x: x if x in label_list else self.masking_value))
                return target_transform(target)

    def __str__(self):
        return f"VOC2012(split={self.split})"

    def __len__(self):
        return len(self.images)

class VOCSegmentationIncrementalMemory(BaseDataset):   
    def __init__(
        self,
        test=False, val=False, setting='overlap', step=0, classes_idx_new=[], classes_idx_old=[], transform=True,
        transform_args={}, masking_value=0, idxs_path=None,
    ):
        if setting not in ['sequential', 'disjoint', 'overlap']:
            raise ValueError('Wrong setting entered! Please use one of sequential, disjoint, overlap')

        super().__init__(
            transform_args=transform_args,
            base_dir=pathlib.Path(DATASETS_IMG_DIRS['voc']),
            transform=transform,
        )
        self.setting = setting

        self.classes_idx_old = classes_idx_old
        self.classes_idx_new = classes_idx_new

        self.test = test
        self.val = val
        self.train = not (self.test or self.val)

        self.masking_value = masking_value

        if self.train:
            self.split = 'train_aug'
        else:
            self.split = 'val'

        if 'aug' not in self.split:
            self._image_dir = self._base_dir / "JPEGImages"
            self._cat_dir = self._base_dir / "SegmentationClass"
        else:
            self._image_dir = self._base_dir
            self._cat_dir = self._base_dir

        self.im_ids = []
        self.images = []
        self.categories = []

        with open(idxs_path, "r") as json_file:
            memory_list = json.load(json_file)

        file_names = memory_list[f"step_{step}"]["memory_list"]
        # new_baseline
        file_xinxi = memory_list[f"step_{step}"]["memory_xinxi"] 
        self.file_xinxi = file_xinxi

        for x in file_names:
            if 'aug' not in self.split:
                _image = self._image_dir / f"{x}.jpg"
                _cat = self._cat_dir / f"{x}.png"
            else:
                _image = self._image_dir / x.split()[0][1:]
                _cat = self._cat_dir / x.split()[1][1:]

            assert _image.is_file(), _image
            assert _image.is_file(), _cat
            
            self.im_ids.append(x)    
            self.images.append(_image) 
            self.categories.append(_cat)    

        assert len(self.images) == len(self.categories)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)

        sample = {"image": _img, "label": _target}

        if self.transform:
            if self.split in ["trainval_aug", "trainval", "train_aug", "train"]:
                sample['image'], sample['label'] = self.transform_tr(sample)
            elif self.split in ["val_aug", "val"]:
                sample['image'], sample['label'] = self.transform_val(sample)
        else:
            sample['image'], sample['label'] = self.transform_test(sample)

        # Target masking
        sample["image_name"] = str(self.im_ids[index])
        file_name = sample["image_name"]
        correct_labels = self.file_xinxi[file_name][0]   
        if 255 in correct_labels:
            correct_labels.remove(255)

        sample['label'] = self.transform_target_masking1(sample['label'], correct_labels)   #new_baseline

       
        unique_classes_y = torch.unique(sample['label'])  
        classes_yt = unique_classes_y[(unique_classes_y != 0) & (unique_classes_y != 255)]
        mask = transformmasks.generate_class_mask(sample['label'],classes_yt) 
        sample['image'] = transformsgpu.instance_replay(mask=mask,data=sample['image'])
        sample['label'] = transformsgpu.instance_replay(mask=mask,data=sample['label'])
        #instance replay finish
        return sample

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert("RGB")
        _target = Image.open(self.categories[index])
        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = tr.Compose(
            [
                tr.RandomResizedCrop(
                    self.transform_args['crop_size'],
                    (0.5, 2.0)
                ),
                tr.RandomHorizontalFlip(),
                tr.ToTensor(),
                tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return composed_transforms(sample['image'], sample['label'])

    def transform_val(self, sample):
        composed_transforms = tr.Compose(
            [
                tr.Resize(size=self.transform_args['crop_size']),
                tr.CenterCrop(self.transform_args['crop_size']),
                tr.ToTensor(),
                tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return composed_transforms(sample['image'], sample['label'])

    def transform_test(self, sample):
        composed_transforms = tr.Compose(
            [
                tr.ToTensor(),
                tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return composed_transforms(sample['image'], sample['label'])

    def transform_target_masking(self, target):
        if self.test:
            # Masking future class object
            # MiB: 255 / PLOP: 0
            label_list = list(set(self.classes_idx_old + self.classes_idx_new + [0, 255]))
            target_transform = tv.transforms.Lambda(
                lambda t: t.apply_(lambda x: x if x in label_list else self.masking_value)
            )
            return target_transform(target)

        else:  # Train or Validation
            # Masking except current classes
            if self.masking_value is None:
                return target
            if self.setting in ['disjoint', 'overlap']:
                label_list = list(set(self.classes_idx_new + [0, 255]))      
                target_transform = tv.transforms.Lambda(
                    lambda t: t.apply_(lambda x: x if x in label_list else self.masking_value)
                )
                return target_transform(target)

    def transform_target_masking1(self, target, labels):
        if self.test:
            # Masking future class object
            # MiB: 255 / PLOP: 0
            label_list = list(set(self.classes_idx_old + self.classes_idx_new + [0, 255]))
            target_transform = tv.transforms.Lambda(
                lambda t: t.apply_(lambda x: x if x in label_list else self.masking_value)
            )
            return target_transform(target)
        else:  # Train or Validation
            # Masking except current classes
            if self.masking_value is None:
                return target
            if self.setting in ['disjoint', 'overlap']:
                label_list = list(set(self.classes_idx_new + [0, 255]))   
                target_transform = tv.transforms.Lambda(lambda t: t.apply_(lambda x: x if x in labels else self.masking_value))     #new_baseline
                return target_transform(target)

    def __len__(self):
        return len(self.images)
    
 
    def get_image_by_name(self, image_name):
  
        try:
            index = self.im_ids.index(image_name)  
        except ValueError:
            raise ValueError(f"Image {image_name} not found in the dataset.")  
        
        #file_name = self.file_names[index]     
        _img, _target = self._make_img_gt_point_pair(index)
        if self.im_ids[index] != image_name: 
            raise ValueError(f"Internal error: Expected file_name to be {image_name}, but got {file_name}.") 
        
        sample = {"image": _img, "label": _target}
        if self.transform:
            if self.split in ["trainval_aug", "trainval", "train_aug", "train"]:
                sample['image'], sample['label'] = self.transform_tr(sample)
            elif self.split in ["val_aug", "val"]:
                sample['image'], sample['label'] = self.transform_val(sample)
        else:
            sample['image'], sample['label'] = self.transform_test(sample)
        labels = self.file_xinxi[image_name][0]


        # return img, target, file_name
        sample['image_name'] = str(self.im_ids[index])
        file_name = sample['image_name']
        correct_labels = self.file_xinxi[file_name][0]    
        if 255 in correct_labels:
            correct_labels.remove(255)

        sample['label'] = self.transform_target_masking1(sample['label'], correct_labels)   #new_baseline

        return sample
