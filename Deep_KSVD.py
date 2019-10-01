"""
Implementation of the Deep K-SVD Denoising model, presented in
Deep K-SVD Denoising
M Scetbon, M Elad, P Milanfar
"""

import os
from skimage import io
import numpy as np
import torch
from torch.utils.data import Dataset


def order_F_to_C(n):
    idx = np.arange(0, n ** 2)
    idx = idx.reshape(n, n, order="F")
    idx = idx.reshape(n ** 2, order="C")
    idx = list(idx)
    return idx


def Init_DCT(n, m):
    """ Compute the Overcomplete Discrete Cosinus Transform. """
    Dictionary = np.zeros((n, m))
    for k in range(m):
        V = np.cos(np.arange(0, n) * k * np.pi / m)
        if k > 0:
            V = V - np.mean(V)
        Dictionary[:, k] = V / np.linalg.norm(V)
    Dictionary = np.kron(Dictionary, Dictionary)
    Dictionary = Dictionary.dot(np.diag(1 / np.sqrt(np.sum(Dictionary ** 2, axis=0))))
    idx = np.arange(0, n ** 2)
    idx = idx.reshape(n, n, order="F")
    idx = idx.reshape(n ** 2, order="C")
    Dictionary = Dictionary[idx, :]
    Dictionary = torch.from_numpy(Dictionary).float()
    return Dictionary


class mydataset_sub_images(Dataset):
    def __init__(self, root_dir, image_names, sub_image_size, sigma, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            image_names (list): List of the images names.
            sub_image_size (integer): Width of the square sub image.
            sigma (float): Level of the noise.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.sub_image_size = sub_image_size
        self.transform = transform
        self.root_dir = root_dir
        self.sigma = sigma
        self.image_names = image_names

        dataset_list = []
        for name in self.image_names:
            np_im = io.imread(os.path.join(self.root_dir, name))
            dataset_list.append(np_im)

        w, h = np.shape(dataset_list[0])
        self.number_sub_images = int(
            (w - sub_image_size + 1) * (h - sub_image_size + 1)
        )

        self.dataset_list = dataset_list
        self.number_images = len(self.image_names)

    def extract_sub_image(self, x, n, k):
        w, h = np.shape(x)
        i = k // int(h - n + 1)
        j = k % int(h - n + 1)
        sub_image = x[i : i + n, j : j + n]
        sub_image = sub_image.reshape(1, n, n)
        return sub_image

    def __len__(self):
        return self.number_images * self.number_sub_images

    def __getitem__(self, idx):

        idx_im = idx // self.number_sub_images
        idx_sub_image = idx % self.number_sub_images

        image = self.dataset_list[idx_im]
        sub_image = self.extract_sub_image(image, self.sub_image_size, idx_sub_image)

        np.random.seed(idx)
        noise = np.random.randn(self.sub_image_size, self.sub_image_size)

        sub_image_noise = sub_image + self.sigma * noise

        if self.transform:
            sub_image = self.transform(sub_image)
            sub_image_noise = self.transform(sub_image_noise)

        return sub_image.float(), sub_image_noise.float()


class mydataset_full_images(Dataset):
    def __init__(self, root_dir, image_names, sigma, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            image_names (list): List of the name of the images.
            sigma (float): Level of the noise.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.transform = transform
        self.root_dir = root_dir
        self.sigma = sigma
        self.image_names = image_names

        dataset_list = []
        dataset_list_noise = []
        for k, name in enumerate(self.image_names):
            np_im = io.imread(os.path.join(self.root_dir, name))
            dataset_list.append(np_im)

            w, h = np.shape(np_im)
            np.random.seed(k + 10 ** 7)
            noise = np.random.randn(w, h)
            np_im_noise = np_im + self.sigma * noise
            dataset_list_noise.append(np_im_noise)

        self.dataset_list_noise = dataset_list_noise
        self.dataset_list = dataset_list
        self.number_images = len(self.image_names)

    def __len__(self):
        return self.number_images

    def __getitem__(self, idx):

        image = self.dataset_list[idx]
        w, h = np.shape(image)
        image = image.reshape(1, w, h)
        image_noise = self.dataset_list_noise[idx]
        image_noise = image_noise.reshape(1, w, h)

        if self.transform:
            image = self.transform(image)
            image_noise = self.transform(image_noise)
        return image.float(), image_noise.float()


class ToTensor(object):
    """ Convert ndarrays to Tensors. """

    def __call__(self, image):
        return torch.from_numpy(image)


class Normalize(object):
    """ Normalize the images. """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        return (image - self.mean) / self.std


class DenoisingNet_MLP(torch.nn.Module):
    def __init__(
        self,
        patch_size,
        D_in,
        H_1,
        H_2,
        H_3,
        D_out_lam,
        T,
        min_v,
        max_v,
        Dict_init,
        c_init,
        w_init,
        device,
    ):

        super(DenoisingNet_MLP, self).__init__()
        self.patch_size = patch_size

        self.T = T
        self.min_v = min_v
        self.max_v = max_v

        q, l = Dict_init.shape
        soft_comp = torch.zeros(l).to(device)
        Identity = torch.eye(l).to(device)

        self.soft_comp = soft_comp
        self.Identity = Identity
        self.device = device

        self.Dict = torch.nn.Parameter(Dict_init)
        self.c = torch.nn.Parameter(c_init)
        self.unfold = torch.nn.Unfold(kernel_size=(self.patch_size, self.patch_size))

        self.linear1 = torch.nn.Linear(D_in, H_1, bias=True)
        self.linear2 = torch.nn.Linear(H_1, H_2, bias=True)
        self.linear3 = torch.nn.Linear(H_2, H_3, bias=True)
        self.linear4 = torch.nn.Linear(H_3, D_out_lam, bias=True)

        self.w = torch.nn.Parameter(w_init)

    def soft_thresh(self, x, l):
        return torch.sign(x) * torch.max(torch.abs(x) - l, self.soft_comp)

    def forward(self, x):

        N, C, w, h = x.shape

        unfold = self.unfold(x)
        N, d, number_patches = unfold.shape
        unfold = unfold.transpose(1, 2)

        lin = self.linear1(unfold).clamp(min=0)
        lin = self.linear2(lin).clamp(min=0)
        lin = self.linear3(lin).clamp(min=0)
        lam = self.linear4(lin)

        l = lam / self.c
        y = torch.matmul(unfold, self.Dict)
        S = self.Identity - (1 / self.c) * self.Dict.t().mm(self.Dict)
        S = S.t()

        z = self.soft_thresh(y, l)
        for t in range(self.T):
            z = self.soft_thresh(torch.matmul(z, S) + (1 / self.c) * y, l)

        x_pred = torch.matmul(z, self.Dict.t())
        x_pred = torch.clamp(x_pred, min=self.min_v, max=self.max_v)
        x_pred = self.w * x_pred
        x_pred = x_pred.transpose(1, 2)

        normalize = torch.ones(N, number_patches, d)
        normalize = normalize.to(self.device)
        normalize = self.w * normalize
        normalize = normalize.transpose(1, 2)

        fold = torch.nn.functional.fold(
            x_pred, output_size=(w, h), kernel_size=(self.patch_size, self.patch_size)
        )

        norm = torch.nn.functional.fold(
            normalize,
            output_size=(w, h),
            kernel_size=(self.patch_size, self.patch_size),
        )

        res = fold / norm

        return res


class DenoisingNet_MLP_2(torch.nn.Module):
    def __init__(
        self,
        patch_size,
        D_in,
        H_1,
        H_2,
        H_3,
        D_out_lam,
        T,
        min_v,
        max_v,
        Dict_init,
        c_init,
        w_1_init,
        w_2_init,
        device,
    ):

        super(DenoisingNet_MLP_2, self).__init__()
        self.patch_size = patch_size

        self.T = T
        self.min_v = min_v
        self.max_v = max_v

        q, l = Dict_init.shape
        soft_comp = torch.zeros(l).to(device)
        Identity = torch.eye(l).to(device)
        self.soft_comp = soft_comp
        self.Identity = Identity
        self.device = device

        self.Dict = torch.nn.Parameter(Dict_init)
        self.c = torch.nn.Parameter(c_init)
        self.unfold = torch.nn.Unfold(kernel_size=(self.patch_size, self.patch_size))

        #### First Stage ####
        self.linear1 = torch.nn.Linear(D_in, H_1, bias=True)
        self.linear2 = torch.nn.Linear(H_1, H_2, bias=True)
        self.linear3 = torch.nn.Linear(H_2, H_3, bias=True)
        self.linear4 = torch.nn.Linear(H_3, D_out_lam, bias=True)

        self.w_1 = torch.nn.Parameter(w_1_init)
        ######################

        #### Second Stage ####
        self.linear1_2 = torch.nn.Linear(D_in, H_1, bias=True)
        self.linear2_2 = torch.nn.Linear(H_1, H_2, bias=True)
        self.linear3_2 = torch.nn.Linear(H_2, H_3, bias=True)
        self.linear4_2 = torch.nn.Linear(H_3, D_out_lam, bias=True)

        self.w_2 = torch.nn.Parameter(w_2_init)
        ######################

    def soft_thresh(self, x, l):
        return torch.sign(x) * torch.max(torch.abs(x) - l, self.soft_comp)

    def forward(self, x):

        N, C, w, h = x.shape

        unfold = self.unfold(x)
        N, d, number_patches = unfold.shape

        unfold = unfold.transpose(1, 2)

        lin = self.linear1(unfold).clamp(min=0)
        lin = self.linear2(lin).clamp(min=0)
        lin = self.linear3(lin).clamp(min=0)
        lam = self.linear4(lin)

        l = lam / self.c
        y = torch.matmul(unfold, self.Dict)
        S = self.Identity - (1 / self.c) * self.Dict.t().mm(self.Dict)
        S = S.t()

        z = self.soft_thresh(y, l)
        for t in range(self.T):
            z = self.soft_thresh(torch.matmul(z, S) + (1 / self.c) * y, l)

        x_pred = torch.matmul(z, self.Dict.t())
        x_pred = torch.clamp(x_pred, min=self.min_v, max=self.max_v)
        x_pred = self.w_1 * x_pred
        x_pred = x_pred.transpose(1, 2)

        normalize = torch.ones(N, number_patches, d)
        normalize = normalize.to(self.device)
        normalize = self.w_1 * normalize
        normalize = normalize.transpose(1, 2)

        fold = torch.nn.functional.fold(
            x_pred, output_size=(w, h), kernel_size=(self.patch_size, self.patch_size)
        )

        norm = torch.nn.functional.fold(
            normalize,
            output_size=(w, h),
            kernel_size=(self.patch_size, self.patch_size),
        )

        res = fold / norm

        ### Second Stage ###
        unfold = self.unfold(res)
        unfold = unfold.transpose(1, 2)

        lin = self.linear1_2(unfold).clamp(min=0)
        lin = self.linear2_2(lin).clamp(min=0)
        lin = self.linear3_2(lin).clamp(min=0)
        lam = self.linear4_2(lin)

        l = lam / self.c
        y = torch.matmul(unfold, self.Dict)

        z = self.soft_thresh(y, l)
        for t in range(self.T):
            z = self.soft_thresh(torch.matmul(z, S) + (1 / self.c) * y, l)

        x_pred = torch.matmul(z, self.Dict.t())
        x_pred = torch.clamp(x_pred, min=self.min_v, max=self.max_v)
        x_pred = self.w_2 * x_pred
        x_pred = x_pred.transpose(1, 2)

        normalize = torch.ones(N, number_patches, d)
        normalize = normalize.to(self.device)
        normalize = self.w_2 * normalize
        normalize = normalize.transpose(1, 2)

        fold = torch.nn.functional.fold(
            x_pred, output_size=(w, h), kernel_size=(self.patch_size, self.patch_size)
        )

        norm = torch.nn.functional.fold(
            normalize,
            output_size=(w, h),
            kernel_size=(self.patch_size, self.patch_size),
        )

        res = fold / norm

        return res


class DenoisingNet_MLP_3(torch.nn.Module):
    def __init__(
        self,
        patch_size,
        D_in,
        H_1,
        H_2,
        H_3,
        D_out_lam,
        T,
        min_v,
        max_v,
        Dict_init,
        c,
        w_1_init,
        w_2_init,
        w_3_init,
        device,
    ):

        super(DenoisingNet_MLP_3, self).__init__()
        self.patch_size = patch_size

        self.T = T
        self.min_v = min_v
        self.max_v = max_v

        q, l = Dict_init.shape
        soft_comp = torch.zeros(l).to(device)
        Identity = torch.eye(l).to(device)
        self.soft_comp = soft_comp
        self.Identity = Identity
        self.device = device

        self.Dict = torch.nn.Parameter(Dict_init)
        self.c = torch.nn.Parameter(c)
        self.unfold = torch.nn.Unfold(kernel_size=(self.patch_size, self.patch_size))

        #### First Stage ####
        self.linear1 = torch.nn.Linear(D_in, H_1, bias=True)
        self.linear2 = torch.nn.Linear(H_1, H_2, bias=True)
        self.linear3 = torch.nn.Linear(H_2, H_3, bias=True)
        self.linear4 = torch.nn.Linear(H_3, D_out_lam, bias=True)

        self.w_1 = torch.nn.Parameter(w_1_init)
        #####################

        #### Second Stage ####
        self.linear1_2 = torch.nn.Linear(D_in, H_1, bias=True)
        self.linear2_2 = torch.nn.Linear(H_1, H_2, bias=True)
        self.linear3_2 = torch.nn.Linear(H_2, H_3, bias=True)
        self.linear4_2 = torch.nn.Linear(H_3, D_out_lam, bias=True)

        self.w_2 = torch.nn.Parameter(w_2_init)
        ######################

        #### Third Stage ####
        self.linear1_3 = torch.nn.Linear(D_in, H_1, bias=True)
        self.linear2_3 = torch.nn.Linear(H_1, H_2, bias=True)
        self.linear3_3 = torch.nn.Linear(H_2, H_3, bias=True)
        self.linear4_3 = torch.nn.Linear(H_3, D_out_lam, bias=True)

        self.w_3 = torch.nn.Parameter(w_3_init)
        ######################

    def soft_thresh(self, x, l):
        return torch.sign(x) * torch.max(torch.abs(x) - l, self.soft_comp)

    def forward(self, x):

        N, C, w, h = x.shape

        unfold = self.unfold(x)
        N, d, number_patches = unfold.shape

        unfold = unfold.transpose(1, 2)

        lin = self.linear1(unfold).clamp(min=0)
        lin = self.linear2(lin).clamp(min=0)
        lin = self.linear3(lin).clamp(min=0)
        lam = self.linear4(lin)

        l = lam / self.c
        y = torch.matmul(unfold, self.Dict)
        S = self.Identity - (1 / self.c) * self.Dict.t().mm(self.Dict)
        S = S.t()

        z = self.soft_thresh(y, l)
        for t in range(self.T):
            z = self.soft_thresh(torch.matmul(z, S) + (1 / self.c) * y, l)

        x_pred = torch.matmul(z, self.Dict.t())
        x_pred = torch.clamp(x_pred, min=self.min_v, max=self.max_v)
        x_pred = self.w * x_pred
        x_pred = x_pred.transpose(1, 2)

        normalize = torch.ones(N, number_patches, d)
        normalize = normalize.to(self.device)
        normalize = self.w_1 * normalize
        normalize = normalize.transpose(1, 2)

        fold = torch.nn.functional.fold(
            x_pred, output_size=(w, h), kernel_size=(self.patch_size, self.patch_size)
        )

        norm = torch.nn.functional.fold(
            normalize,
            output_size=(w, h),
            kernel_size=(self.patch_size, self.patch_size),
        )

        res = fold / norm

        ### Second Stage ###
        unfold = self.unfold(res)
        unfold = unfold.transpose(1, 2)

        lin = self.linear1_2(unfold).clamp(min=0)
        lin = self.linear2_2(lin).clamp(min=0)
        lin = self.linear3_2(lin).clamp(min=0)
        lam = self.linear4_2(lin)

        l = lam / self.c
        y = torch.matmul(unfold, self.Dict)

        z = self.soft_thresh(y, l)
        for t in range(self.T):
            z = self.soft_thresh(torch.matmul(z, S) + (1 / self.c) * y, l)

        x_pred = torch.matmul(z, self.Dict.t())
        x_pred = torch.clamp(x_pred, min=self.min_v, max=self.max_v)
        x_pred = self.w_2 * x_pred
        x_pred = x_pred.transpose(1, 2)

        normalize = torch.ones(N, number_patches, d)
        normalize = normalize.to(self.device)
        normalize = self.w_2 * normalize
        normalize = normalize.transpose(1, 2)

        fold = torch.nn.functional.fold(
            x_pred, output_size=(w, h), kernel_size=(self.patch_size, self.patch_size)
        )

        norm = torch.nn.functional.fold(
            normalize,
            output_size=(w, h),
            kernel_size=(self.patch_size, self.patch_size),
        )

        res = fold / norm

        ### Third Stage ###
        unfold = self.unfold(res)
        unfold = unfold.transpose(1, 2)

        lin = self.linear1_3(unfold).clamp(min=0)
        lin = self.linear2_3(lin).clamp(min=0)
        lin = self.linear3_3(lin).clamp(min=0)
        lam = self.linear4_3(lin)

        l = lam / self.c
        y = torch.matmul(unfold, self.Dict)

        z = self.soft_thresh(y, l)
        for t in range(self.T):
            z = self.soft_thresh(torch.matmul(z, S) + (1 / self.c) * y, l)

        x_pred = torch.matmul(z, self.Dict.t())
        x_pred = torch.clamp(x_pred, min=self.min_v, max=self.max_v)
        x_pred = self.w_3 * x_pred
        x_pred = x_pred.transpose(1, 2)

        normalize = torch.ones(N, number_patches, d)
        normalize = normalize.to(self.device)
        normalize = self.w_3 * normalize
        normalize = normalize.transpose(1, 2)

        fold = torch.nn.functional.fold(
            x_pred, output_size=(w, h), kernel_size=(self.patch_size, self.patch_size)
        )

        norm = torch.nn.functional.fold(
            normalize,
            output_size=(w, h),
            kernel_size=(self.patch_size, self.patch_size),
        )

        res = fold / norm

        return res
