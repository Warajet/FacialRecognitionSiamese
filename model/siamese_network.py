import torch
import torchvision.models as models
import torch.nn.functional as F


class SiameseNetwork(torch.nn.Module):
    def __init__(self, embeddings_dim, margin = 1.0):
        """
        Initialize the neural network
        """
        super(SiameseNetwork, self).__init__()

        self.margin = margin
        self.embeddings_dim = embeddings_dim

        # Load pre-trained ResNet-18 as the base model
        self.pretrained_resnet = models.resnet18(pretrained=True)

        # Remove the last fully connected layer (classification head)
        self.pretrained_resnet = torch.nn.Sequential(*list(self.pretrained_resnet.children())[:-1])

        # Freeze the ResNet layers
        for param in self.pretrained_resnet.parameters():
            param.requires_grad = False

        # Fully connected layers for embedding
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, self.embeddings_dim)
        )

    def forward_once(self, image):
        out = self.pretrained_resnet(image)
        out = out.view(-1, 512)
        out = self.fc(out)
        return out
    
    def forward(self, anchor, positive, negative):
        """
        Define the behaviour of the forward pass
        """
        anchor_embeddings = self.forward_once(anchor)
        positive_embeddings = self.forward_once(positive)
        negative_embeddings = self.forward_once(negative)

        return anchor_embeddings, positive_embeddings, negative_embeddings
    
    def get_embedding(self, image):
        out = self.forward_once(image)
        return out
