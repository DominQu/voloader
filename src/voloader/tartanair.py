import numpy as np
import cv2
from torch.utils.data import Dataset
import torch

from os import listdir
from pathlib import Path
from tqdm import tqdm

from .transformation import pos_quats2SEs, pose2motion, SEs2ses
from .utils import make_intrinsics_layer


class TrajectoryBatchSampler:
    def __init__(self, traj_ranges):
        self.traj_ranges = traj_ranges
        self.num_traj = len(self.traj_ranges) - 1

    def __iter__(self):
        order = range(self.num_traj)
        for k in order:
            start, end = self.traj_ranges[k], self.traj_ranges[k+1]
            yield list(range(start, end))

    def __len__(self):
        return self.num_traj

class TartanDataset(Dataset):
    def __init__(self, 
                 data_path: str,
                 train: bool = True,
                 combined: bool = True,
                 transform = None,
                 modality = "all",
                 focalx = 320.0, 
                 focaly = 320.0, 
                 centerx = 320.0, 
                 centery = 240.0,
                 std = None):
        """Base class for TartanAir dataset.
        Args:
            data_path (str): Path to the dataset folder.
            train (bool): If True, expect TartanAir training directory structure; otherwise, test directory structure.
            combined (bool): If True, combine all trajectories into common lists; otherwise, keep them separate.
            transform (callable, optional): A function/transform that takes in a sample and returns a transformed version.
            modality (str): Type of data to load. One of: all, img, flow.
            focalx (float): Focal length in x direction.
            focaly (float): Focal length in y direction.
            centerx (float): X coordinate of the image center.
            centery (float): Y coordinate of the image center.
            std (list): List with std for each element of motion vector.
        """
        
        self.N = 0
        self.index_ranges = [0]
        self.data_path = Path(data_path)
        if train:
            self.trajectories = sorted(list(self.data_path.glob('*/*/*')))
        else:
            self.trajectories = sorted(list(self.data_path.glob('*')))
        self.combined = combined
        self.std = std
        self.dataset = self._load_data(self.trajectories, self.combined)
        self.transform = transform
        if modality not in ["all", "img", "flow"]:
            raise ValueError(f"Provided modality: {modality} not supported.")
        self.modality = modality

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
            dict: Dictionary containing nested dictonaries with images, flows, and relative poses from/for each trajectory.
                  If combined is True, all trajectories are combined into a single nested dictionary under the key 'combined'.
        """
        
        print("Building TartanAir dataset")
        
        if combined:
            dataset = {'combined':
                {
                'images': [],
                'flows': [],
                'relposes': []
                }
            }
        else:
            dataset = {}

        for traj in tqdm(trajectories):
            images = sorted(traj.glob('image_left/*.png'))
            flows = sorted(traj.glob('flow/*flow.npy'))

            assert len(images) == len(flows) + 1, \
            "The number of flow files should be one less than the number of image files. " \
            f"Found {len(images)} images and {len(flows)} flow files in {traj}."

            poselist = np.loadtxt(traj / "pose_left.txt").astype(np.float32)
            assert(poselist.shape[1]==7) # position + quaternion
            poses = pos_quats2SEs(poselist)
            matrix = pose2motion(poses)
            motions = SEs2ses(matrix).astype(np.float32)
            if self.std is not None:
                motions = motions / np.array(self.std).reshape((1, -1))
            assert(len(motions) == len(images)) - 1

            if combined:
                dataset['combined']['images'].extend(images)
                dataset['combined']['flows'].extend(flows)
                dataset['combined']['relposes'].extend(motions)
            else:
                dataset[traj] = {'images': images, 'flows': flows, 
                    'relposes': motions}
                self.index_ranges.append(self.index_ranges[-1] + len(flows))
            self.N += len(flows)

        if not combined:
            assert self.index_ranges[-1] == self.N, \
            "The upper index range should match the total number of items in the dataset."
            self.index_ranges = np.array(self.index_ranges)
        
        return dataset
    
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        if self.combined:
            traj_name = 'combined'
            sample_idx = idx
        else:
            # If not combined, find the trajectory that contains the index
            mask = self.index_ranges < idx
            cummask = np.cumsum(mask)
            traj_idx = np.argmax(cummask)
            traj_name = self.trajectories[traj_idx]
            sample_idx = idx - self.index_ranges[traj_idx]
        
        if traj_name not in self.dataset:
            raise ValueError(f"Trajectory {traj_name} not found in dataset.")
        res = {}
        if self.modality in ["all", "img"]:
            imgfile1 = self.dataset[traj_name]['images'][sample_idx]
            imgfile2 = self.dataset[traj_name]['images'][sample_idx + 1]
            img1 = cv2.imread(imgfile1)
            img2 = cv2.imread(imgfile2)
            res['img1'] = img1
            res['img2'] = img2
            h, w, _ = img1.shape
            intrinsicLayer = make_intrinsics_layer(w, h, self.focalx, self.focaly, self.centerx, self.centery)
            res['intrinsic'] = intrinsicLayer  
        if self.modality in ["all", "flow"]:
            flowfile = self.dataset[traj_name]['flows'][sample_idx]
            flow = np.load(flowfile)
            res['flow'] = flow
            # Add intrinsic layer for flow modality
            if not "intrinsic" in list(res.keys()):
                h, w, _ = flow.shape
                intrinsicLayer = make_intrinsics_layer(w, h, self.focalx, self.focaly, self.centerx, self.centery)
                res['intrinsic'] = intrinsicLayer 

        if self.transform:
            res = self.transform(res)
        res['relpose'] = torch.tensor(self.dataset[traj_name]['relposes'][sample_idx], dtype=torch.float32)
            
        return res

# class TartanImgPoseDataset(TartanDataset):
#     """Tartan dataset providing only images and poses"""
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#     def __getitem__(self, idx):
#         if self.combined:
#             raise ValueError("Combined mode not supported for image dataset")
#             # traj_name = 'combined'
#             # sample_idx = idx
#         else:
#             # If not combined, find the trajectory that contains the index
#             mask = self.index_ranges <= idx
#             cummask = np.cumsum(mask)
#             traj_idx = np.argmax(cummask)
#             traj_name = self.trajectories[traj_idx]
#             sample_idx = idx - self.index_ranges[traj_idx]
        
#         if traj_name not in self.dataset:
#             raise ValueError(f"Trajectory {traj_name} not found in dataset.")
        
#         imgfile1 = self.dataset[traj_name]['images'][sample_idx]
#         imgfile2 = self.dataset[traj_name]['images'][sample_idx + 1]
#         img1 = cv2.imread(imgfile1)
#         img2 = cv2.imread(imgfile2)
        
#         res = {'img': np.concat([img1, img2], axis=-1)}

#         res['relpose'] = torch.tensor(self.dataset[traj_name]['relposes'][sample_idx], dtype=torch.float32)
            
#         if self.transform is not None:
#             res['img'] = self.transform(res['img'])
#         return res

# class TartanFlowPoseDataset(TartanDataset):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def __getitem__(self, idx):
#         if self.combined:
#             traj_name = 'combined'
#             sample_idx = idx
#         else:
#             # If not combined, find the trajectory that contains the index
#             mask = self.index_ranges < idx
#             cummask = np.cumsum(mask)
#             traj_idx = np.argmax(cummask)
#             traj_name = self.trajectories[traj_idx]
#             sample_idx = idx - self.index_ranges[traj_idx]
        
#         if traj_name not in self.dataset:
#             raise ValueError(f"Trajectory {traj_name} not found in dataset.")

#         flowfile = self.dataset[traj_name]['flows'][sample_idx]
#         flow = np.load(flowfile)
#         res = {}
#         res['flow'] = flow

#         res['relpose'] = self.dataset[traj_name]['relposes'][sample_idx]
#         h, w, _ = flow.shape
#         intrinsicLayer = make_intrinsics_layer(w, h, self.focalx, self.focaly, self.centerx, self.centery)
#         res['intrinsic'] = intrinsicLayer 
#         if self.transform:
#             res = self.transform(res)

#         return res

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

