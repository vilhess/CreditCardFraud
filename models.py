import torch 
import torch.nn as nn 

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.seq = nn.Sequential(
            nn.Linear(30, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.seq(x)
    

class Projector(nn.Module):
    def __init__(self, proj_dim=32):
        super(Projector, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(30, 128, bias=False),
            nn.LeakyReLU(),
            nn.Linear(128, proj_dim, bias=False)
        )
    def forward(self, x):
        return self.seq(x)

class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dims, latent_dim):
        super(Encoder, self).__init__()

        self.seq = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(hidden_dims[1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[1], latent_dim)

    def forward(self, x):
        h = self.seq(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, out_dim):
        super(Decoder, self).__init__()

        self.seq = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], out_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        reconstructed = self.seq(z)
        return reconstructed
    

class VAE(nn.Module):
    def __init__(self, in_dim, hidden_dims, latent_dim):
        super(VAE, self).__init__()

        self.encoder = Encoder(in_dim=in_dim, hidden_dims=hidden_dims, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, hidden_dims=hidden_dims[::-1], out_dim=in_dim)

    def _rep_trick(self, mu, logvar):
        epsilon = torch.randn_like(logvar)
        return mu + torch.sqrt(logvar.exp())*epsilon
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self._rep_trick(mu, logvar)
        out = self.decoder(z)
        return out, mu, logvar