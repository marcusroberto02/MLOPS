"""
LFW dataloading
"""
import argparse
import time

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform) -> None:
        # TODO: fill out with what you need
        self.transform = transform
        with open('data/names.txt') as topo_file:
            imgs = []
            for line in topo_file:
                name, count = line.split()
                for i in range(int(count)):
                    idx = str(i+1)
                    img = Image.open(f"{path_to_folder}/{name}/{name}_{idx.zfill(4)}.jpg")
                    imgs.append(img)
        self.data = imgs
        
    def __len__(self):
        return len(self.data) # TODO: fill out
    
    def __getitem__(self, index: int) -> torch.Tensor:
        # TODO: fill out
        img = self.data[index]
        return self.transform(img)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_folder', default='data/lfw', type=str)
    parser.add_argument('-batch_size', default=512, type=int)
    parser.add_argument('-num_workers', default=1, type=int)
    parser.add_argument('-visualize_batch', action='store_true')
    parser.add_argument('-get_timing', action='store_true')
    parser.add_argument('-get_timing_errorplot', action='store_true')
    parser.add_argument('-batches_to_check', default=100, type=int)
    
    args = parser.parse_args()
    
    lfw_trans = transforms.Compose([
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)
    
    # Define dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )
    
    if args.visualize_batch:
        # TODO: visualize a batch of images
        imgperrow = 3
        fig, axs = plt.subplots(nrows=imgperrow,ncols=imgperrow,squeeze=False)
        for i in range(imgperrow):
            for j in range(imgperrow):
                img = dataset[i*imgperrow+j].detach()
                img = F.to_pil_image(img)
                axs[i, j].imshow(np.asarray(img))
                axs[i, j].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.show()
        
    if args.get_timing:
        # lets do some repetitions
        res = [ ]
        for _ in range(5):
            start = time.time()
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx > args.batches_to_check:
                    break
            end = time.time()

            res.append(end - start)
            
        res = np.array(res)
        print(f'Timing: {np.mean(res)}+-{np.std(res)}')
    
    if args.get_timing_errorplot:
        times = []
        errors = []
        num_workers = 8
        for n in tqdm(range(1,num_workers+1)):
            dataloader = DataLoader(
                dataset, 
                batch_size=args.batch_size, 
                shuffle=False,
                num_workers=n
            )
            res = [ ]
            for _ in range(5):
                start = time.time()
                for batch_idx, batch in enumerate(dataloader):
                    if batch_idx > args.batches_to_check:
                        break
                end = time.time()

                res.append(end - start)
                
            res = np.array(res)
            times.append(np.mean(res))
            errors.append(np.std(res))

        workers = range(1,num_workers+1)
        plt.errorbar(workers,times,errors)
        plt.show()


