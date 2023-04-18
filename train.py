#import sklearn
#from sklearn import datasets
#from sklearn.model_selection import train_test_split

#import mlflow

#X, y = datasets.load_iris(return_X_y=True)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1337)

#knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)
#knn.fit(X_train, y_train)

#knn.score(X_test, y_test)

import os

import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

# For brevity, here is the simplest most minimal example with just a training
# loop step, (no validation, no testing). It illustrates how you can use MLflow
# to auto log parameters, metrics, and models.


class MNISTModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=1)
        acc = self.accuracy(pred, y)

        # Use the current of PyTorch logger
        #self.log("train_loss", loss, on_epoch=True)
        #self.log("acc", acc, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


# Initialize our model
mnist_model = MNISTModel()

# Initialize DataLoader from MNIST Dataset
train_ds = MNIST(
    os.getcwd(), train=True, download=True, transform=transforms.ToTensor()
)
train_loader = DataLoader(train_ds, batch_size=32)

# Initialize a trainer
trainer = pl.Trainer(max_epochs=2)

# Train the model
trainer.fit(mnist_model, train_loader)
