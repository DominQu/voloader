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
                 train: bool = True,
                 combined: bool = True,
                 transform=None,
                 std=None):
        """KITTI Odometry Dataset.

        Args:
            data_path (str): Root KITTI odometry directory.
            sequences (list[str]): Sequence IDs (e.g. ["00", "01"]). If None, use all.
            train (bool): Unused, for API compatibility.
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
                }
            }
        else:
            dataset = {}

        for seq in tqdm(self.sequences):
            seq_dir = self.seq_path / seq / "image_2"
            images = sorted(seq_dir.glob("*.png"))

            pose_file = self.pose_path / f"{seq}.txt"
            poses = np.loadtxt(pose_file).reshape(-1, 3, 4).astype(np.float32)
            matrix = pose2motion(poses)
            motions = SEs2ses(matrix).astype(np.float32)
            # Convert to 4x4
            # poses_4x4 = np.zeros((poses.shape[0], 4, 4), dtype=np.float32)
            # poses_4x4[:, :3, :4] = poses
            # poses_4x4[:, 3, 3] = 1.0

            # # Relative motions
            # relposes = []
            # for i in range(len(poses_4x4) - 1):
            #     T1 = poses_4x4[i]
            #     T2 = poses_4x4[i + 1]
            #     rel = np.linalg.inv(T1) @ T2
            #     relposes.append(rel)

            # relposes = np.stack(relposes)

            # # Convert SE(3) -> se(3)
            # relposes = SEs2ses(relposes).astype(np.float32)

            if self.std is not None:
                motions /= np.array(self.std).reshape(1, -1)

            assert len(images) == len(motions) + 1

            if self.combined:
                dataset["combined"]["images"].extend(images)
                dataset["combined"]["relposes"].extend(motions)
            else:
                dataset[seq] = {
                    "images": images,
                    "relposes": motions,
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
            traj_idx = np.searchsorted(self.index_ranges, idx, side="right") - 1
            seq_name = self.sequences[traj_idx]
            sample_idx = idx - self.index_ranges[traj_idx]

        imgfile1 = self.dataset[seq_name]["images"][sample_idx]
        imgfile2 = self.dataset[seq_name]["images"][sample_idx + 1]

        img1 = cv2.imread(str(imgfile1))
        img2 = cv2.imread(str(imgfile2))

        h, w, _ = img1.shape
        intrinsic = make_intrinsics_layer(
            w, h, self.focalx, self.focaly, self.centerx, self.centery
        )

        res = {
            "img": np.concat([img1, img2], axis=-1),
            "intrinsic": intrinsic,
            "relpose": torch.tensor(
                self.dataset[seq_name]["relposes"][sample_idx],
                dtype=torch.float32,
            ),
        }

        if self.transform:
            # Transform only the image
            res["img"] = self.transform(res["img"])

        return res
