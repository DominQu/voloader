import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from os import listdir
from .transformation import pos_quats2SEs, pose2motion, SEs2ses
from .utils import make_intrinsics_layer

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


