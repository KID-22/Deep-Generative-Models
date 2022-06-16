import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, data_dim, hidden_dim, z_dim):
        super().__init__()

        self.MLP = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, data_dim),
        )

        self.apply(weights_init)

    def forward(self, z):
        return self.MLP(z)


class Discriminator(nn.Module):
    def __init__(self, data_dim, hidden_dim):
        super().__init__()

        self.MLP = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(weights_init)

    def forward(self, x):
        return self.MLP(x).view(-1)


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)


class WGAN(nn.Module):
    def __init__(self, data_dim=2, hidden_dim=400, z_dim=2, device="cuda"):
        super().__init__()
        self.__dict__.update(locals())

        self.G = Generator(data_dim, hidden_dim, z_dim)
        self.D = Discriminator(data_dim, hidden_dim)

    def calculate_D_loss(self, z, real_data):
        """ Calculate D loss """
        fake_data = self.G(z).detach()  # only update D
        fake_data_score = self.D(fake_data)
        real_data_score = self.D(real_data)
        D_loss = torch.mean(fake_data_score) - torch.mean(real_data_score)
        return D_loss

    def calculate_G_loss(self, z):
        """ Calculate G loss """
        fake_data = self.G(z)
        fake_data_score = self.D(fake_data)
        G_loss = -torch.mean(fake_data_score)
        return G_loss

    def evaluate(self, valid_dataloader):
        valid_G_loss, valid_D_loss = [], []
        for bacth_data in valid_dataloader:
            real_data = bacth_data.to(self.device)
            z = torch.randn(len(bacth_data), self.z_dim, device=self.device)
            batch_D_loss = self.calculate_D_loss(z, real_data)
            batch_G_loss = self.calculate_G_loss(z)

            valid_G_loss.append(batch_G_loss.item())
            valid_D_loss.append(batch_D_loss.item())

        return valid_G_loss, valid_D_loss

    def clip_paras(self, clip_value):
        for p in self.D.parameters():
            p.data.clamp_(-clip_value, clip_value)

    def get_optimizer(self, lr):
        optimizer_G = torch.optim.RMSprop(self.G.parameters(), lr=lr)
        optimizer_D = torch.optim.RMSprop(self.D.parameters(), lr=lr)
        return optimizer_G, optimizer_D

    def save_model(self, savepath):
        """ Save model state dictionary """
        torch.save(self.G.state_dict(), savepath + "_G.pth")
        torch.save(self.D.state_dict(), savepath + "_D.pth")

    def load_model(self, loadpath):
        """ Load state dictionary into model """
        self.G.load_state_dict(torch.load(loadpath + "_G.pth"))
        self.D.load_state_dict(torch.load(loadpath + "_D.pth"))