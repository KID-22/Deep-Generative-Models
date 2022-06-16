import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import numpy as np
from model.WGAN import WGAN
from utils import DGM_Data, save_image


class WGAN_Config(object):
    dataset = "circle_1"

    data_dim = 2
    hidden_dim = 400
    z_dim = 2

    batch_size = 2048
    lr = 5e-5
    clip_value = 0.1

    max_epoch = 50000
    verbose = 1000

    device = "cuda"
    seed_num = 2022


def WGAN_Train(opt):
    # get data
    train_data = DGM_Data(filename="../data/" + opt.dataset + "/train.txt")
    valid_data = DGM_Data(filename="../data/" + opt.dataset + "/valid.txt")
    test_data = DGM_Data(filename="../data/" + opt.dataset + "/test.txt")

    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=opt.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=True)

    model = WGAN(data_dim=opt.data_dim, hidden_dim=opt.hidden_dim, z_dim=opt.z_dim, device=opt.device)
    model.to(opt.device)
    optimizer_G, optimizer_D = model.get_optimizer(opt.lr)

    full_train_G_loss = []
    full_train_D_loss = []
    full_valid_G_loss = []
    full_valid_D_loss = []

    for epoch in range(1, opt.max_epoch+1):
        model.train()
        train_G_loss, train_D_loss = [], []

        for bacth_data in train_dataloader:
            # 1. train discriminator for k steps
            for _ in range(5):
                real_data = bacth_data.to(opt.device)
                z = torch.randn(len(bacth_data), opt.z_dim, device=opt.device)
                batch_D_loss = model.calculate_D_loss(z, real_data)
                optimizer_D.zero_grad()
                batch_D_loss.backward()
                optimizer_D.step()

                train_D_loss.append(batch_D_loss.item())

                model.clip_paras(opt.clip_value)

            # 2. train generator
            z = torch.randn(opt.batch_size, opt.z_dim, device=opt.device)
            batch_G_loss = model.calculate_G_loss(z)
            optimizer_G.zero_grad()
            batch_G_loss.backward()
            optimizer_G.step()

            train_G_loss.append(batch_G_loss.item())

        # save epoch loss
        full_train_G_loss.append(np.mean(train_G_loss))
        full_train_D_loss.append(np.mean(train_D_loss))

        # # evaluation
        model.eval()
        valid_G_loss, valid_D_loss = model.evaluate(valid_dataloader)
        full_valid_G_loss.append(np.mean(valid_G_loss))
        full_valid_D_loss.append(np.mean(valid_D_loss))

        # save images
        if epoch % opt.verbose == 0:
            # randomly sample some data to visualize
            z = torch.randn(len(valid_data), opt.z_dim).to(opt.device)
            model.eval()
            sample_data = model.G(z).detach().cpu().numpy()
            save_image(dataset_name=opt.dataset, model_name="WGAN", epoch=epoch, real_data=valid_data, sample_data=sample_data)

        if epoch % 10 ==0:
            print("Epoch[%d/%d]\n**Train**: G Loss: %.4f, D Loss: %.4f"
                % (epoch, opt.max_epoch, np.mean(train_G_loss),
                np.mean(train_D_loss)))
            print("Epoch[%d/%d]\n**Valid**: G Loss: %.4f, D Loss: %.4f"
                % (epoch, opt.max_epoch, np.mean(valid_G_loss),
                np.mean(valid_D_loss)))

    # save model
    model.save_model("../ckpt/" + opt.dataset + "/WGAN/best_model")

    # # save loss
    np.savetxt("../result/" + opt.dataset + "/WGAN/full_train_G_loss.txt", np.array(full_train_G_loss))
    np.savetxt("../result/" + opt.dataset + "/WGAN/full_train_D_loss.txt", np.array(full_train_D_loss))
    np.savetxt("../result/" + opt.dataset + "/WGAN/full_valid_G_loss.txt", np.array(full_valid_G_loss))
    np.savetxt("../result/" + opt.dataset + "/WGAN/full_valid_D_loss.txt", np.array(full_valid_D_loss))

    print("\n=============best model=============")
    model.load_model("../ckpt/" + opt.dataset + "/WGAN/best_model")
    model.eval()
    G_loss, D_loss = model.evaluate(train_dataloader)
    print("**Train**: G Loss: %.4f, D Loss: %.4f" % (np.mean(G_loss), np.mean(D_loss)))
    G_loss, D_loss = model.evaluate(valid_dataloader)
    print("**Valid**: G Loss: %.4f, D Loss: %.4f" % (np.mean(G_loss), np.mean(D_loss)))
    G_loss, D_loss = model.evaluate(test_dataloader)
    print("**Test**: G Loss: %.4f, D Loss: %.4f" % (np.mean(G_loss), np.mean(D_loss)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument('--dataset', default='circle_1', choices=["circle_1", "circle_2"], type=str)

    args = parser.parse_args()

    opt = WGAN_Config()
    opt.dataset = args.dataset

    print('\n'.join(['%s:%s' % item for item in opt.__dict__.items()]))

    os.makedirs("../result/" + opt.dataset + "/WGAN/images/", exist_ok=True)
    os.makedirs("../ckpt/" + opt.dataset + "/WGAN/", exist_ok=True)

    WGAN_Train(opt)