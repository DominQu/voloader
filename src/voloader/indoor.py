import numpy as np
import cv2
from torch.utils.data import Dataset
import torch

from os import listdir
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from .transformation import pos_quats2SEs, pose2motion, SEs2ses
from .utils import make_intrinsics_layer

def load_gt_for_images(images, gt_file):
    # Extract timestamps from image filenames (keep order)
    img_timestamps = [
        os.path.splitext(os.path.basename(p))[0] for p in images
    ]

    # Load ground truth file
    df = pd.read_csv(
        gt_file,
        sep=r"\s+",
        header=None,
        names=["timestamp","p1","p2","p3","p4","p5","p6","p7"]
    )

    # Convert timestamps to string to match filenames
    df["timestamp"] = df["timestamp"].astype(str)

    # Keep only rows corresponding to loaded images
    df = df[df["timestamp"].isin(img_timestamps)]
    # Reorder to match the order of images
    df = df.set_index("timestamp").reindex(img_timestamps)
    print("Nans in df:", pd.isna(df).sum(axis=0))
    print(df.dropna().head())
    # Return numpy array of poses (N x 7)
    df = df.dropna()
    poses = df[["p1","p2","p3","p4","p5","p6","p7"]].to_numpy()
    parent = Path(images[0]).parent
    images = [parent / f"{timestamp}.jpg" for timestamp in df.index]
    return poses, images

def load_associations(images, file):
    # Extract timestamps from image filenames (keep order)
    img_timestamps = [
        os.path.splitext(os.path.basename(p))[0] for p in images
    ]

    # Load ground truth file
    df = pd.read_csv(
        file,
        sep=r"\s+",
        header=None,
        names=["timestamp", "path", "label_timestamp","p1","p2","p3","p4","p5","p6","p7"]
    )

    # Convert timestamps to string to match filenames
    for col in ["timestamp", "path", "label_timestamp"]:
        df[col] = df[col].astype(str)

    # Keep only rows corresponding to loaded images
    df = df[df["timestamp"].isin(img_timestamps)]
    # Reorder to match the order of images
    df = df.set_index("timestamp").reindex(img_timestamps)
    print("Nans in df:", pd.isna(df).sum(axis=0))
    print(df.dropna().head())
    # Return numpy array of poses (N x 7)
    df = df.dropna()
    poses = df[["p1","p2","p3","p4","p5","p6","p7"]].to_numpy()
    parent = Path(images[0]).parent.parent
    images = [parent / f"{path}" for path in df["path"]]
    return poses, images

class IndoorDataset(Dataset):
    def __init__(self, 
                 data_path: str,
                 transform = None,
                 focalx = 1470.0142, 
                 focaly = 1470.0142, 
                 centerx = 958.5608, 
                 centery = 722.47815,
                 tumrgbd = True,
                 std = None):
        """Base class for TartanAir dataset.
        Args:
            data_path (str): Path to the dataset folder.
            transform (callable, optional): A function/transform that takes in a sample and returns a transformed version.
            focalx (float): Focal length in x direction.
            focaly (float): Focal length in y direction.
            centerx (float): X coordinate of the image center.
            centery (float): Y coordinate of the image center.
            std (list): List with std for each element of motion vector.
        """
        
        self.N = 0
        self.index_ranges = [0]
        self.data_path = Path(data_path)

        self.std = std
        self.dataset = self._load_data()
        self.transform = transform
        if tumrgbd:
            self.focalx = 525.0  # focal length x
            self.focaly = 525.0  # focal length y
            self.centerx = 319.5  # optical center x
            self.centery = 239.5
        else:
            self.focalx = focalx
            self.focaly = focaly
            self.centerx = centerx
            self.centery = centery
    
    def _load_data(self) -> dict:
        """Load data from path."""
        
        print("Building Indoor dataset")
        

        dataset = {}
        images = sorted((self.data_path / "rgb" ).glob("*.png"))
        print(f"Found {len(images)} images")
        # Make images the same len as flows by combining paths that are following each other
        poselist, images = load_associations(images, self.data_path / "associations.txt")
        # poselist, images = load_gt_for_images(images, self.data_path / "groundtruth.txt")
        print(f"Found {len(poselist)} GT poses")
        assert(poselist.shape[1]==7) # position + quaternion
        images = [[images[i], images[i+1]] for i in range(len(images)-1)]
        
        poses = pos_quats2SEs(poselist)
        matrix = pose2motion(poses)
        motions = SEs2ses(matrix).astype(np.float32)
        # motions[:,:3] = motions[:, :3] / np.linalg.norm(motions[:,:3], axis=1)[..., None]
        if self.std is not None:
            motions = motions / np.array(self.std).reshape((1, -1))

        assert(len(motions) == len(images)), \
        "The number of relative poses should be equal to the number of image pairs. " \
        f"Found {len(images)} image pairs and {len(motions)} relative poses."
        

        dataset["images"] = images
        dataset["relposes"] = motions
        
        self.N = len(images)
        
        return dataset
    
    def __len__(self):
        return self.N

    def __getitem__(self, idx):


        sample_idx = idx

        res = {}

        imgfile1 = self.dataset['images'][sample_idx][0]
        imgfile2 = self.dataset['images'][sample_idx][1]
        img1 = cv2.imread(imgfile1)
        img2 = cv2.imread(imgfile2)
        res['img1'] = img1
        res['img2'] = img2
        h, w, _ = img1.shape
        intrinsicLayer = make_intrinsics_layer(w, h, self.focalx, self.focaly, self.centerx, self.centery)
        res['intrinsic'] = intrinsicLayer  

        if self.transform:
            res = self.transform(res)
        res['relpose'] = torch.tensor(self.dataset['relposes'][sample_idx], dtype=torch.float32)
        return res


