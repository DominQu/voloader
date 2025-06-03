import numpy as np
import cv2
from torch.utils.data import Dataset

from os import listdir
from pathlib import Path
from tqdm import tqdm

from .transformation import pos_quats2SEs, pose2motion, SEs2ses
from .utils import make_intrinsics_layer

class TartanDataset(Dataset):
    def __init__(self, 
                 data_path: str,
                 train: bool = True,
                 combined: bool = True,
                 transform = None,
                 focalx = 320.0, 
                 focaly = 320.0, 
                 centerx = 320.0, 
                 centery = 240.0):
        """Base class for TartanAir dataset.
        Args:
            data_path (str): Path to the dataset folder.
            train (bool): If True, expect TartanAir training directory structure; otherwise, test directory structure.
            combined (bool): If True, combine all trajectories into common lists; otherwise, keep them separate.
            transform (callable, optional): A function/transform that takes in a sample and returns a transformed version.
            focalx (float): Focal length in x direction.
            focaly (float): Focal length in y direction.
            centerx (float): X coordinate of the image center.
            centery (float): Y coordinate of the image center.
        """
        
        self.N = 0
        self.data_path = Path(data_path)
        if train:
            trajectories = list(data_path.glob('*/*/*'))
        else:
            trajectories = list(data_path.glob('*'))
        self.combined = combined
        self.dataset = self._load_data(trajectories, self.combined)
        self.transform = transform
        self.focalx = focalx
        self.focaly = focaly
        self.centerx = centerx
        self.centery = centery
    
    def _load_data(self, trajectories: list[Path], combined: bool = True) -> dict:
        """Load data from paths in trajectories list.
        Args:
            trajectories (list): List of paths to trajectories.
            combined (bool): If True, combine all trajectories into common lists.
        Returns:
            dict: Dictionary containing images, flows, and poses from/for each trajectory.
        """
        
        print("Building TartanAir dataset")
        
        if combined:
            dataset = {
                'images': [],
                'flows': [],
                'poses': []
            }
        else:
            dataset = {}

        for traj in tqdm(sorted(trajectories)):
            images = sorted(traj.glob('image_left/*.png'))
            flows = sorted(traj.glob('flow/*flow.npy'))

            assert len(images) == len(flows) + 1, \
            "The number of flow files should be one less than the number of image files."

            poselist = np.loadtxt(traj / "pose_left.txt").astype(np.float32)
            assert(poselist.shape[1]==7) # position + quaternion
            poses = pos_quats2SEs(poselist)
            matrix = pose2motion(poses)
            motions = SEs2ses(matrix).astype(np.float32)
            # FUTURE: consider normalizing the motions
            assert(len(motions) == len(images)) - 1

            if combined:
                dataset['images'].extend(images)
                dataset['flows'].extend(flows)
                dataset['poses'].extend(motions)
            else:
                dataset[traj] = {'images': images, 'flows': flows, 
                    'poses': poses}
            self.N += len(flows)

        return dataset
    
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        if not self.combined:
            raise NotImplementedError("This method is not implemented for non-combined datasets.")
        
        imgfile1 = self.dataset['images'][idx]
        imgfile2 = self.dataset['images'][idx+1]
        img1 = cv2.imread(imgfile1)
        img2 = cv2.imread(imgfile2)
        
        res = {'img1': img1, 'img2': img2}

        h, w, _ = img1.shape
        intrinsicLayer = make_intrinsics_layer(w, h, self.focalx, self.focaly, self.centerx, self.centery)
        res['intrinsic'] = intrinsicLayer  

        flowfile = self.flowfiles[idx]
        flow = np.load(flowfile).transpose((2, 0, 1)) if flowfile is not None else None
        res['flow'] = flow

        if self.motions is not None:
            res['motion'] = self.motions[idx]
            
        if self.transform:
            res = self.transform(res)
        
        return res

class TrajFolderDataset(Dataset):
    """scene flow synthetic dataset. """

    def __init__(self, imgfolder , posefile = None, transform = None, flowdir = None,
                    focalx = 320.0, focaly = 320.0, centerx = 320.0, centery = 240.0):
        
        files = listdir(imgfolder)
        self.rgbfiles = [(imgfolder +'/'+ ff) for ff in files if (ff.endswith('.png') or ff.endswith('.jpg'))]
        self.rgbfiles.sort()
        self.imgfolder = imgfolder

        print('Found {} image files in {}'.format(len(self.rgbfiles), imgfolder))

        if flowdir is not None:
            flow_files = listdir(flowdir)
            self.flowfiles = [(flowdir +'/'+ ff) for ff in flow_files if ff.endswith('flow.npy')]
            self.flowfiles.sort()
            assert len(self.rgbfiles) == len(self.flowfiles) + 1, "The number of flow files should be one less than the number of image files."
        else:
            self.flowfiles = [None] * (len(self.rgbfiles) - 1)
        
        if posefile is not None and posefile!="":
            poselist = np.loadtxt(posefile).astype(np.float32)
            assert(poselist.shape[1]==7) # position + quaternion
            poses = pos_quats2SEs(poselist)
            self.matrix = pose2motion(poses)
            self.motions     = SEs2ses(self.matrix).astype(np.float32)
            # self.motions = self.motions / self.pose_std
            assert(len(self.motions) == len(self.rgbfiles)) - 1
        else:
            self.motions = None

        self.N = len(self.rgbfiles) - 1

        self.transform = transform
        self.focalx = focalx
        self.focaly = focaly
        self.centerx = centerx
        self.centery = centery

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        imgfile1 = self.rgbfiles[idx].strip()
        imgfile2 = self.rgbfiles[idx+1].strip()
        img1 = cv2.imread(imgfile1)
        img2 = cv2.imread(imgfile2)
        
        res = {'img1': img1, 'img2': img2}

        h, w, _ = img1.shape
        intrinsicLayer = make_intrinsics_layer(w, h, self.focalx, self.focaly, self.centerx, self.centery)
        res['intrinsic'] = intrinsicLayer

        if self.transform:
            res = self.transform(res)
        
        flowfile = self.flowfiles[idx]
        flow = np.load(flowfile).transpose((2, 0, 1)) if flowfile is not None else None
        res['flow'] = flow

        if self.motions is None:
            return res
        else:
            res['motion'] = self.motions[idx]
            return res

