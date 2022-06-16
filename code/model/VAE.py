import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    MLP encoder for VAE.
    input: data
    output: mu and log_(sigma**2)
    """
    def __init__(self, data_dim, hidden_dim, z_dim):
        super().__init__()

        self.MLP = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
        )
        self.mu = nn.Linear(hidden_dim, z_dim) # mu
        self.log_sigma = nn.Linear(hidden_dim, z_dim) # log(sigma**2)


    def forward(self, x):
        out = self.MLP(x)
        mu, log_sigma = self.mu(out), self.log_sigma(out)
        return mu, log_sigma


class Decoder(nn.Module):
    """ 
    MLP decoder for VAE.
    input: z
    output: reconstructed data
    """
    def __init__(self, z_dim, hidden_dim, data_dim):
        super().__init__()

        self.MLP = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
        )
        self.reconstruct = nn.Linear(hidden_dim, data_dim)

    def forward(self, z):
        out = self.MLP(z)
        rec = self.reconstruct(out)
        return rec


class VAE(nn.Module):
    """
    VAE model.
    """
    def __init__(self, data_dim=2, hidden_dim=400, z_dim=2, device='cuda'):
        super().__init__()
        self.__dict__.update(locals())

        self.encoder = Encoder(data_dim=data_dim, hidden_dim=hidden_dim, z_dim=z_dim)
        self.decoder = Decoder(z_dim=z_dim, hidden_dim=hidden_dim, data_dim=data_dim)

        self.device = device

    def forward(self, x):
        mu, log_sigma = self.encoder(x)
        z = self.reparameterize(mu, log_sigma)
        rec = self.decoder(z)
        return rec, mu, log_sigma

    def reparameterize(self, mu, log_sigma):
        """"
        Reparametrization trick: z = mu + epsilon * sigma, where epsilon ~ N(0, 1).
        """
        std = torch.exp(log_sigma * 0.5)
        eps = torch.randn_like(std)
        return mu + std * eps

    def get_optimizer(self, lr, weight_decay):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def calculate_loss(self, inputs):
        """
        Calculate loss for VAE.
        """
        outputs, mu, log_sigma = self.forward(inputs)

        # 1. reconstruction loss
        rec_loss = torch.sum((inputs - outputs) ** 2)

        # 2. KL divergence loss
        kl_loss = self.kl_divergence(mu, log_sigma)

        total_loss = rec_loss + kl_loss

        return total_loss, rec_loss, kl_loss

    def kl_divergence(self, mu, log_sigma):
        """ Compute Kullback-Leibler divergence """
        return torch.sum(0.5 * (mu**2 + torch.exp(log_sigma) - log_sigma - 1))

    def evaluate(self, valid_dataloader):
        valid_loss, valid_rec_loss, valid_kl_loss = [], [], []
        for bacth_data in valid_dataloader:
            inputs = bacth_data.to(self.device)
            batch_loss, batch_rec_loss, batch_kl_loss = self.calculate_loss(inputs)

            valid_loss.append(batch_loss.item())
            valid_rec_loss.append(batch_rec_loss.item())
            valid_kl_loss.append(batch_kl_loss.item())

        return valid_loss, valid_rec_loss, valid_kl_loss

    def save_model(self, savepath):
        """ Save model state dictionary """
        torch.save(self.state_dict(), savepath)

    def load_model(self, loadpath):
        """ Load state dictionary into model """
        state = torch.load(loadpath)
        self.load_state_dict(state)
