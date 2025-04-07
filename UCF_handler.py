# Read the test video and break it down into frames
import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class UCF101VideoDataset(Dataset):
    def __init__(self, video_folder, transform=None, start=0, end=10):
        self.video_folder = video_folder
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((64, 64)),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
        self.start = start
        self.end = end
        self.video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith('.avi')]

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        frames = self.load_video(video_path)
        selected_frames = frames[self.start:self.end]  # Extract frames within a specific range
        if self.transform:
            # use transform frame by frame
            selected_frames = torch.stack([self.transform(frame) for frame in selected_frames], dim=0)
        return selected_frames

    def load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return np.array(frames)

