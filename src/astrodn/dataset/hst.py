"""PyTorch Dataset for Hubble Space Telescope images."""

import glob
import os
from urllib import request

import numpy as np
from torchvision.datasets import VisionDataset
from tqdm import tqdm

from astrodn.utils import loadfits


class HSTDataset(VisionDataset):
    """Dataset for Hubble Space Telescope images."""

    URL_TEMPLATE = (
        "https://hla.stsci.edu/cgi-bin/getdata.cgi?config=ops&dataset={file}"
    )

    def __init__(
        self,
        root: str,
        patchsize: int,
        *args,
        train=True,
        download=False,
        transforms=None,
        **kwargs,
    ):
        """Instantiates a Hubble Space Telescope (HST) dataset.

        Args:
            root (str):
                Root directory to store dataset files
            patchsize (int): 
                Size of patches that will be fed to the models.
            train (bool): 
                `True` for training dataset, otherwise use test dataset.
            download (bool): 
                `True` to download files. `False` to disable downloads.
            transform (callable, optional):
                Transformations to apply to images.
        
        Returns:
            HSTDataset
                A pytorch dataset
        """
        super().__init__(root, transforms, *args, **kwargs)
        self._patchsize = patchsize

        self._ds = self._read_dataset(train)
        if not self._dataset_exists(train) and download:
            self._download(train)

    @property
    def files(self):
        """File paths for each dataset entry."""
        return [path for path, _ in self._ds]

    def _dataset_exists(self, train: bool):
        if not os.path.exists(self.root):
            return False
        
        basedir = os.path.join(self.root, "train" if train else "test")
        files = glob.glob(os.path.join(basedir, "*.fits"))

        return len(self._ds) <= len(files)


    def _read_dataset(self, train: bool) -> list[tuple[str, str]]:
        """Read dataset entries.

        Entries are stored in either 'dataset-train.txt' or
        'dataset-test.txt' in the same directory as this
        source file. The train flag indicates which file to
        read entries from.

        Args:
            train: Read train dataset if True

        Returns:
            List of tuples (path, url), where `path` is the path
            of the file on disk and `url` is the URL to download
            the file.
        """
        pwd = os.path.dirname(os.path.realpath(__file__))
        file = os.path.join(
            pwd, "dataset-train.txt" if train else "dataset-test.txt"
        )

        ds = []
        with open(file) as f:
            for entry in f.read().splitlines():
                # ignore commented lines
                if entry.startswith("#"):
                    continue

                url = self.URL_TEMPLATE.format(file=entry)
                path = os.path.join(
                    self.root,
                    "train" if train else "test",
                    f"{entry}_drz.fits",
                )
                ds.append((path, url))

        return ds

    def _download(self, train: bool):
        """Download dataset files.

        Args:
            train: Download train files.
        """
        dest = os.path.join(self.root, "train" if train else "test")
        if not os.path.exists(dest):
            os.makedirs(dest)

        for path, url in self._ds:
            with request.urlopen(url) as res:
                total = int(res.getheader("Content-Length", 0))

            with tqdm(total=total, unit="B", unit_scale=True, desc=path) as t:

                def progress(block_num, block_size, total_size):
                    t.update(block_size)

                request.urlretrieve(url, path, progress)

    def _random_patch(self, data: np.ndarray):
        w, h = data.shape
        while True:
            x = np.random.randint(0, w - self._patchsize + 1)
            y = np.random.randint(0, h - self._patchsize + 1)
            patch = data[x : x + self._patchsize, y : y + self._patchsize]

            if self._zero_ratio(patch) < 0.4:
                return patch

    def _zero_ratio(self, patch: np.ndarray):
        zeros = patch[patch == 0].size
        return zeros / patch.size

    def __getitem__(self, index: int):
        """Get image pair from dataset.

        Args:
            index: Entry index.

        Returns:
            Image tuple (input, target).
        """
        path, _ = self._ds[index]

        data = loadfits(path)
        patch = self._random_patch(data)

        if self.transforms:
            return self.transforms(patch, patch)
        return patch, patch

    def __len__(self):
        """Dataset length.

        Returns:
            Dataset length.
        """
        return len(self._ds)
