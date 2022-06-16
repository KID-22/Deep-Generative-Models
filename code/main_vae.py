import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import numpy as np
from model.VAE import VAE
from utils import DGM_Data, save_image


class VAE_Config(object):
    dataset = "circle_1"

    data_dim = 2
    hidden_dim = 400
    z_dim = 2

    batch_size = 2048
    lr = 1e-3
    weight_decay = 1e-5

    max_epoch = 50000
    verbose = 1000

    device = "cuda"
    seed_num = 2022


def VAE_Train(opt):
    # get data
    train_data = DGM_Data(filename="../data/" + opt.dataset + "/train.txt")
    valid_data = DGM_Data(filename="../data/" + opt.dataset + "/valid.txt")
    test_data = DGM_Data(filename="../data/" + opt.dataset + "/test.txt")

    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=opt.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=True)

    model = VAE(data_dim=opt.data_dim, hidden_dim=opt.hidden_dim, z_dim=opt.z_dim, device=opt.device)
    model.to(opt.device)
    optimizer = model.get_optimizer(opt.lr, opt.weight_decay)

    best_valid_loss = 1e10
    best_epoch = 0

    full_train_loss = []
    full_train_rec_loss = []
    full_train_kl_loss = []
    full_valid_loss = []
    full_valid_rec_loss = []
    full_valid_kl_loss = []

    for epoch in range(1, opt.max_epoch+1):
        model.train()
        train_loss, train_rec_loss, train_kl_loss = [], [], []

        for bacth_data in train_dataloader:
            inputs = bacth_data.to(opt.device)
            batch_loss, batch_rec_loss, batch_kl_loss = model.calculate_loss(inputs)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            train_loss.append(batch_loss.item())
            train_rec_loss.append(batch_rec_loss.item())
            train_kl_loss.append(batch_kl_loss.item())

        # save epoch loss
        full_train_loss.append(np.sum(train_loss)/len(train_data))
        full_train_rec_loss.append(np.sum(train_rec_loss)/len(train_data))
        full_train_kl_loss.append(np.sum(train_kl_loss)/len(train_data))

        # evaluation
        model.eval()
        valid_loss, valid_rec_loss, valid_kl_loss = model.evaluate(valid_dataloader)
        valid_loss_avg = np.sum(valid_loss)/len(valid_data)
        full_valid_loss.append(np.sum(valid_loss)/len(valid_data))
        full_valid_rec_loss.append(np.sum(valid_rec_loss)/len(valid_data))
        full_valid_kl_loss.append(np.sum(valid_kl_loss)/len(valid_data))

        # save best model
        if valid_loss_avg < best_valid_loss:
            best_valid_loss, best_epoch = valid_loss_avg, epoch
            model.save_model("../ckpt/" + opt.dataset + "/VAE/best_model.pth")

        # save images
        if epoch % opt.verbose == 0:
            # randomly sample some data to visualize
            z = torch.randn(len(valid_data), opt.z_dim).to(opt.device)
            model.eval()
            sample_data = model.decoder(z).detach().cpu().numpy()
            save_image(dataset_name=opt.dataset, model_name="VAE", epoch=epoch, real_data=valid_data, sample_data=sample_data)

        if epoch % 10 ==0:
            print("Epoch[%d/%d]\n**Train**: Total Loss: %.4f, Reconst Loss: %.4f, KL Div: %.4f"
                % (epoch, opt.max_epoch, np.sum(train_loss)/len(train_data),
                np.sum(train_rec_loss)/len(train_data), np.sum(train_kl_loss)/len(train_data)))
            print("Epoch[%d/%d]\n**Valid**: Total Loss: %.4f, Reconst Loss: %.4f, KL Div: %.4f"
                % (epoch, opt.max_epoch, np.sum(valid_loss)/len(valid_data),
                np.sum(valid_rec_loss)/len(valid_data), np.sum(valid_kl_loss)/len(valid_data)))
            print("Best Epoch[%d/%d]\n**Valid**: Total Loss: %.4f"
                % (best_epoch, opt.max_epoch, best_valid_loss))

    # save loss
    np.savetxt("../result/" + opt.dataset + "/VAE/full_train_loss.txt", np.array(full_train_loss))
    np.savetxt("../result/" + opt.dataset + "/VAE/full_train_rec_loss.txt", np.array(full_train_rec_loss))
    np.savetxt("../result/" + opt.dataset + "/VAE/full_train_kl_loss.txt", np.array(full_train_kl_loss))
    np.savetxt("../result/" + opt.dataset + "/VAE/full_valid_loss.txt", np.array(full_valid_loss))
    np.savetxt("../result/" + opt.dataset + "/VAE/full_valid_rec_loss.txt", np.array(full_valid_rec_loss))
    np.savetxt("../result/" + opt.dataset + "/VAE/full_valid_kl_loss.txt", np.array(full_valid_kl_loss))

    print("\n=============best model=============")
    model.load_model("../ckpt/" + opt.dataset + "/VAE/best_model.pth")
    model.eval()
    loss, rec_loss, kl_loss = model.evaluate(train_dataloader)
    print("Best Epoch[%d/%d]\n**Train**: Total Loss: %.4f, Reconst Loss: %.4f, KL Div: %.4f"
                % (best_epoch, opt.max_epoch, np.sum(loss)/len(train_data),
                np.sum(rec_loss)/len(train_data), np.sum(kl_loss)/len(train_data)))
    loss, rec_loss, kl_loss = model.evaluate(valid_dataloader)
    print("Best Epoch[%d/%d]\n**Valid**: Total Loss: %.4f, Reconst Loss: %.4f, KL Div: %.4f"
                % (best_epoch, opt.max_epoch, np.sum(loss)/len(valid_data),
                np.sum(rec_loss)/len(valid_data), np.sum(kl_loss)/len(valid_data)))
    loss, rec_loss, kl_loss = model.evaluate(test_dataloader)
    print("Best Epoch[%d/%d]\n**Test**: Total Loss: %.4f, Reconst Loss: %.4f, KL Div: %.4f"
                % (best_epoch, opt.max_epoch, np.sum(loss)/len(test_data),
                np.sum(rec_loss)/len(test_data), np.sum(kl_loss)/len(test_data)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument('--dataset', default='circle_1', choices=["circle_1", "circle_2"], type=str)

    args = parser.parse_args()

    opt = VAE_Config()
    opt.dataset = args.dataset

    print('\n'.join(['%s:%s' % item for item in opt.__dict__.items()]))

    os.makedirs("../result/" + opt.dataset + "/VAE/images/", exist_ok=True)
    os.makedirs("../ckpt/" + opt.dataset + "/VAE/", exist_ok=True)

    VAE_Train(opt)