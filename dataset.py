import os
import cv2

class KITTIDataset:
    def __init__(self, image_dir):
        self.image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.endswith(".png")
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
