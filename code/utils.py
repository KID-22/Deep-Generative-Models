from torch.utils import data
import numpy as np
import torch
import copy
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False # 解决坐标轴负数的负号显示问题


class DGM_Data(data.Dataset):
    def __init__(self, filename):
        raw_matrix = np.loadtxt(filename, dtype='float32')
        self.x = int(max(raw_matrix[:, 0]))
        self.y = int(max(raw_matrix[:, 1]))
        self.data = raw_matrix

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]


def save_image(dataset_name, model_name, epoch, real_data, sample_data):
    plt.figure(1,figsize=(6,6)) 
    plt.scatter(real_data[:,0], real_data[:,1], c="orange", marker=".", s=1, label="Real")
    plt.scatter(sample_data[:,0], sample_data[:,1], c="green", marker="+", s=1, alpha=0.5, label="Fake")

    # plt.title('Distribution of Real data and Fake data from' + model_name, fontsize=17)
    plt.title(model_name, fontsize=17)
    plt.xlabel('x', fontsize=15)
    plt.ylabel('y', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # plt.xlim(-78, 78)
    # plt.ylim(-78, 78)
    plt.rcParams.update({'font.size': 15})
    plt.legend()

    plt.tight_layout()
    plt.savefig("../result/%s/%s/images/%d.jpg" % (dataset_name, model_name, epoch) , dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
