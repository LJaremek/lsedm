from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import pandas as pd
import numpy as np
import torch


class ImageNet(Dataset):
    def __init__(
            self,
            split: str,
            classes: list[int] | None = None,
            n_samples: int | None = None,
            cache_dir: str | None = None
            ) -> None:
        """
        Inputs:
        * spit: str - 'train' or 'validation'
        * classes: list[int] | None - list of classes. Available classes:
            https://huggingface.co/datasets/huggingface/label-files/blob/main/imagenet-1k-id2label.json
        * n_samples: int | None = None - number of samples per class.
            Default value is None (all records from the class)
        """
        super().__init__()
        assert split in ("train", "validation")

        self.img_size = 256
        self.split = split
        self.n_samples = n_samples
        self.classes = classes

        self._limit_dataset(split, n_samples, classes, cache_dir)
        self.label_names = self.data.features["label"]

        self.transform = transforms.Compose([
            transforms.CenterCrop(256),
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple:
        """
        The input 'index' is the index of record in the ImageNet class instance
        The output index is the original image index from the entire dataset
        """
        original_index = self.filtered_indices[index]
        item = self.data[original_index]
        img = item["image"]
        label = item["label"]

        # img = transforms.Compose([transforms.Resize((self.img_size, self.img_size))])(img)
        # img = transforms.Compose([transforms.ToTensor()])(img)
        # img = transforms.Compose([transforms.Normalize((0.5,), (0.5,))])(img)

        # img_transformed = self._openai_transform(img)
        # labels = self._get_labels_dict(label)

        # return img_transformed, labels, original_index
        return {
            "img": self.transform(img),
            "index": original_index,
            "labels": label
            }

    def _get_labels_dict(self, label: int) -> dict[int, int]:
        """
        Returns a dictionary with labels for a single image.
        The keys are the label names (ints) and the values are 1 or 0.
        """
        labels_dict = {
            self.label_names.int2str(i): (1 if i == label else 0)
            for i in range(1000)
            }
        return labels_dict

    def _limit_dataset(
            self,
            split: str,
            n_samples: int | None = None,
            classes: list[str] | None = None,
            cache_dir: str | None = None
            ) -> None:
        """
        ### Example where classes=(1, 2) and n_samples=1

        Input data (self.data) contains:
         * image: A, B, C, D, E, F
         * label: 1, 0, 2, 1, 0, 2

        Because classes contains (1, 2) so gt_labels looks like:
         * image: A, C, D, F
         * label: 1, 2, 1, 2

        Then self.filtered_indices looks like:
         * filtered_index: 0, 1, 2, 3
         * original_index: 0, 2, 3, 5

        If n_samples is provided, it limits the number of records per class.
        For example, if n_samples=1:
         * image: A, C
         * label: 1, 2

        Then self.filtered_indices looks like:
         * filtered_index: 0, 1
         * original_index: 0, 2
        """
        if cache_dir is None:
            cache_dir = "~/.cache/huggingface/datasets"

        self.data = load_dataset(
            "imagenet-1k",
            split=split,
            trust_remote_code=True,
            cache_dir=cache_dir
            )

        gt_labels = pd.DataFrame(self.data["label"], columns=["gt_label"])
        print("LEN 1:", len(gt_labels))  # TODO: delete

        if classes is None:
            self.filtered_indices = gt_labels.index.tolist()
        else:
            self.filtered_indices = gt_labels[
                gt_labels["gt_label"].isin(classes)
                ].index.tolist()
        print("LEN 2:", len(self.filtered_indices))  # TODO: delete

        if n_samples is not None and classes is not None:
            limited_indices = []
            for class_label in classes:
                class_indices = gt_labels[
                    gt_labels["gt_label"] == class_label
                    ].index.tolist()
                limited_indices.extend(class_indices[:n_samples])
            self.filtered_indices = limited_indices

        print("LEN 3:", len(self.filtered_indices))  # TODO: delete

        self.length = len(self.filtered_indices)

    def _openai_transform(self, img):
        """
        Ported from openai/improved-diffusion
        """
        while min(*img.size) >= 2 * self.img_size:
            img = img.resize(
                tuple(x // 2 for x in img.size), resample=Image.BOX)

        scale = self.img_size / min(*img.size)
        img = img.resize(
            tuple(round(x * scale) for x in img.size), resample=Image.BICUBIC)

        arr = np.array(img.convert("RGB"))
        crop_y = (arr.shape[0] - self.img_size) // 2
        crop_x = (arr.shape[1] - self.img_size) // 2
        arr = arr[
            crop_y: crop_y + self.img_size,
            crop_x: crop_x + self.img_size
            ]
        arr = arr.astype(np.float32) / 255.

        img = np.transpose(arr, [2, 0, 1])

        return torch.from_numpy(img)
