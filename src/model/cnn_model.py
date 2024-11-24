# Defining CNN model from the paper
import torch
import torch.nn as nn



# CNN model from paper
class StressCNN(nn.Module):
    def __init__(self, num_classes) -> None:
        """
        Initializing the model
        """
        
        super(StressCNN, self).__init__()

        self.num_classes = num_classes

        # Convolutional part of the model for feature learning
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 3), stride=1, padding='same'),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(0.3),
            nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(0.3)
        )

        self.flatten = nn.Flatten()
        self.classifier = None



    def forward(self, x) -> torch.Tensor:
        """
        Forward pass through the model
        """
        x = self.features(x)

        # Classifier part 
        if self.classifier is None:
            num_features = x.size(1) * x.size(2) * x.size(3)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(num_features, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, self.num_classes)
            )
            # Move "classifier" part to the same device where rest of the model is
            self.classifier.to(x.device)

        x = self.flatten(x)
        x = self.classifier(x)
        return x
    