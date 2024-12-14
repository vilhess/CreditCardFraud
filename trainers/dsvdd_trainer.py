import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import trange, tqdm

class DeepSVDD:
    def __init__(self, model, proj_dim, device, eps=0.1):
        """
        Classe pour l'entraînement du modèle DeepSVDD.
        
        :param model: Le modèle PyTorch.
        :param proj_dim: Dimension de projection.
        :param device: Appareil (CPU ou GPU).
        :param eps: Écart minimal pour éviter des centres proches de 0.
        """
        self.model = model
        self.proj_dim = proj_dim
        self.device = device
        self.eps = eps
        self.center = None

    def initialize_center(self, trainloader):
        """
        Initialisation du centre `c`.
        
        :param trainloader: DataLoader contenant les données d'entraînement.
        :return: Le centre initial `c`.
        """
        n_samples = 0
        c = torch.zeros(self.proj_dim).to(self.device)

        self.model.eval()
        with torch.no_grad():
            for x, _ in tqdm(trainloader, desc="Initialisation du centre"):
                x = x.to(self.device)
                outputs = self.model(x)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)
        c /= n_samples

        c[(abs(c) < self.eps) & (c < 0)] = -self.eps
        c[(abs(c) < self.eps) & (c > 0)] = self.eps

        self.model.train()
        self.center = c
        return c

    def train(self, trainloader, epochs, optimizer: Optimizer, scheduler: _LRScheduler):
        """
        Entraîne le modèle DeepSVDD.
        
        :param trainloader: DataLoader contenant les données d'entraînement.
        :param epochs: Nombre d'époques.
        :param optimizer: Optimiseur PyTorch.
        :param scheduler: Ordonnanceur de taux d'apprentissage.
        :return: Un dictionnaire contenant le modèle entraîné et le centre.
        """
        if self.center is None:
            raise ValueError("Le centre n'a pas été initialisé. Appelez `initialize_center` avant d'entraîner.")

        pbar = range(epochs)
        for epoch in pbar:
            curr_loss = 0
            for x, y in tqdm(trainloader, desc="Training"):
                x = x.to(self.device)
                optimizer.zero_grad()
                projects = self.model(x)
                dist = torch.sum((projects - self.center) ** 2, dim=1)
                loss = torch.mean(dist)
                curr_loss += loss.item()
                
                loss.backward()
                optimizer.step()

            scheduler.step()
            print(f"Epoch {epoch + 1}/{epochs} ; Loss: {curr_loss / len(trainloader):.4f}")

        return {"model": self.model, "center": self.center}