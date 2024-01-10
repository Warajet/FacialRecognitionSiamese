from torch.utils.data import Dataset, DataLoader
import torch
from utils import generate_samples

class FaceTripletDataset(Dataset):
    def __init__(self, triplets, transform = None):
        self.triplets = generate_samples(triplets)
        self.transform = transform
    
    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        triplet = self.triplets[index]
        
        anchor = triplet[0]
        positive = triplet[1]
        negative = triplet[2]

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        sample = {"anchor": anchor, "positive": positive, "negative": negative}
        return sample
    