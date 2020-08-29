import random
import PIL
from PIL import Image

import pydicom
import numpy as np
import os

import torch
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    """Simple input, target classification"""
    def __init__(self, X, y, transform=None):
        """
        Arguments:
            X (sequence): the training examples
            y (sequence): the training targets
            transform (callable, optional): transform applied on training 
                examples
        """
        assert len(X) == len(y), f"Mismatch in number of instances X ({len(X)}) and y ({len(y)})"
        self.X = X
        self.y = y
        self.transform = transform
        
    def __getitem__(self, key):
        x, y = self.X[key], self.y[key]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.X)


class Segmentation2DDataset(Dataset):
    """
    Dataset for segmentation tasks
    Supports 2D DICOM format (.dcm) and common image formats (.png, .jpg)
    """

    def __init__(self, image_pairs, input_transform=None,
                 mask_transform=None, input_image_handler=None,
                 mask_image_handler=None, cache=False):
        """
        Arguments:
            image_pairs (sequence): sequence of (input, mask) image pairs. Can
                either be pairs of filepaths for the images or 
            input_transform (callable, optional): the transform to be applied 
                on input images
            mask_transform (callable, optional): the transform to be applied 
                on target/mask images
            input_image_handler (callable, optional): the handler to open
                input images. By default, the input file's extension is 
                used to select the appropriate handler
            mask_image_handler (callable, optional): the handler to open
                mask images. By default, the mask file's extension is 
                used to select the appropriate handler
            cache (bool, optional): if True, will load all images to memory 
                if image pairs are filepaths. By default will load the images
                lazily
        """
        assert len(image_pairs) > 0, \
            f"Expected non empty sequence for input target pairs"

        if isinstance(image_pairs[0], np.ndarray) \
            or isinstance(image_pairs[0], torch.Tensor):
            self.are_image_paths = False

        else:
            self.are_image_paths = True
        
        self.image_pairs = image_pairs
        self.input_transform = input_transform
        self.mask_transform = mask_transform
        self.input_image_handler = input_image_handler
        self.mask_image_handler = mask_image_handler
        self.cache = cache
        self.seed = np.random.randint(2147483647)

        if cache and self.are_image_paths:
            self._cache_segmentation_pairs()

    def _cache_segmentation_pairs(self):
        """Load all input image and target images to memory"""
        self.cached_segmentation_pairs = []
        for input_image_fp, mask_image_fp in self.image_pairs:

            input_image = self._load_image_array(input_image, 
                self.input_image_handler)
            mask_image = self._load_image_array(mask_image_fp,
                self.mask_image_handler)

            self.cached_segmentation_pairs.append((input_image, mask_image))

    def _load_image_array(self, image_fp, handler=None):
        """Load the image as an array"""
        _, file_extension = os.path.splitext(image_fp)

        if handler:
            return handler(image_fp)

        if file_extension == ".dcm":
            dicom_obj = pydicom.dcmread(image_fp)
            image_array = dicom_obj.pixel_array
        else:
            image_array = np.array(Image.open(image_fp).load())

        return image_array

    def __getitem__(self, key):

        if self.cache and self.are_image_paths:
            input_image, mask_image = self.cached_segmentation_pairs[key]

        elif not self.cache and self.are_image_paths:

            input_image_fp, mask_image_fp = self.image_pairs[key]

            input_image = self._load_image_array(input_image_fp, 
                self.input_image_handler)
            mask_image = self._load_image_array(mask_image_fp,
                self.mask_image_handler)

        else:
            input_image, mask_image = self.image_pairs[key]
        
        if self.input_transform and self.mask_transform:

            # Need to use the same seed for the random package, so that any
            # random properties for both input and target transforms are the same
            random.seed(self.seed)
            torch.manual_seed(self.seed)

            input_image = self.input_transform(input_image)
            mask_image = self.mask_transform(mask_image)

        elif self.input_transform:
            input_image = self.input_transform(input_image)
            
        elif self.mask_transform:
            mask_image = self.mask_transform(mask_image)

        return input_image, mask_image

    def __len__(self):
        return len(self.image_pairs)
