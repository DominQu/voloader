from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import torch
import cv2
from tqdm import tqdm

from .utils import make_intrinsics_layer, dataset_intrinsics
from .transformation import SEs2ses, pose2motion


class KITTIOdometryDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 sequences=None,
                 combined: bool = False,
                 transform=None,
                 std=None,
                 **kwargs):
        """KITTI Odometry Dataset.

        Args:
            data_path (str): Root KITTI odometry directory.
            sequences (list[str]): Sequence IDs (e.g. ["00", "01"]). If None, use all.
            combined (bool): Combine all sequences or keep separate.
            transform (callable): Optional transform.
            std (list): Optional normalization for motion vector.
        """

        self.N = 0
        self.index_ranges = [0]
        self.data_path = Path(data_path)
        self.seq_path = self.data_path / "sequences"
        self.pose_path = self.data_path / "poses"

        if sequences is None:
            sequences = sorted([p.name for p in self.seq_path.iterdir() if p.is_dir()])
        else:
            sequences = [str(seq) for seq in sequences]

        self.sequences = sequences
        self.combined = combined
        self.transform = transform
        self.std = std

        self.focalx, self.focaly, self.centerx, self.centery = dataset_intrinsics(dataset='kitti')

        self.dataset = self._load_data()

    def _load_data(self):
        print("Building KITTI Odometry dataset")

        if self.combined:
            dataset = {
                "combined": {
                    "images": [],
                    "relposes": [],
                    "flows": []
                }
            }
        else:
            dataset = {}

        for seq in tqdm(self.sequences):
            seq_dir = self.seq_path / seq / "image_2"
            images = sorted(seq_dir.glob("*.png"))
            
            flow_dir = self.seq_path / seq / "pred_flow"
            if Path(flow_dir).is_dir():
                flows = sorted(flow_dir.glob('*flow.npy'))

                assert len(images)-1 == len(flows), \
                "The number of image pairs should be equal to number of flows. " \
                f"Found {len(images)} image pairs and {len(flows)} flow files in {flow_dir}."
            else:
                print(f"Didn't find any flow for trajectory {flow_dir}")
                flows = None

            pose_file = self.pose_path / f"{seq}.txt"
            poses = np.loadtxt(pose_file).reshape(-1, 3, 4).astype(np.float32)
            matrix = pose2motion(poses)
            motions = SEs2ses(matrix).astype(np.float32)

            if self.std is not None:
                motions /= np.array(self.std).reshape(1, -1)

            assert len(images) == len(motions) + 1

            if self.combined:
                dataset["combined"]["images"].extend(images)
                dataset["combined"]["relposes"].extend(motions)
                dataset["combined"]["flows"].extend(flows)
            else:
                dataset[seq] = {
                    "images": images,
                    "relposes": motions,
                    "flows": flows
                }
                self.index_ranges.append(self.index_ranges[-1] + len(motions))

            self.N += len(motions)

        if not self.combined:
            self.index_ranges = np.array(self.index_ranges)
            assert self.index_ranges[-1] == self.N

        return dataset

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        if self.combined:
            seq_name = "combined"
            sample_idx = idx
        else:
            # traj_idx = np.searchsorted(self.index_ranges, idx, side="right") - 1
            # seq_name = self.sequences[traj_idx]
            # sample_idx = idx - self.index_ranges[traj_idx]
            # If not combined, find the trajectory that contains the index
            mask = self.index_ranges <= idx
            cummask = np.cumsum(mask)
            traj_idx = np.argmax(cummask)
            seq_name = self.sequences[traj_idx]
            sample_idx = idx - self.index_ranges[traj_idx]


        imgfile1 = self.dataset[seq_name]["images"][sample_idx]
        imgfile2 = self.dataset[seq_name]["images"][sample_idx + 1] # This will work for last frames, because sample_idx will always be one less than the number of images
        
        img1 = cv2.imread(str(imgfile1))
        img2 = cv2.imread(str(imgfile2))

        flowfile = self.dataset[seq_name]['flows'][sample_idx]
        flow = np.load(flowfile)

        h, w, _ = img1.shape
        intrinsic = make_intrinsics_layer(
            w, h, self.focalx, self.focaly, self.centerx, self.centery
        )

        res = {
            "img1": img1,
            "img2": img2,
            "flow": flow,
            "intrinsic": intrinsic
        }

        if self.transform:
            res = self.transform(res)
        res["relpose"]= torch.tensor(
                self.dataset[seq_name]["relposes"][sample_idx],
                dtype=torch.float32,
            )
        return res
