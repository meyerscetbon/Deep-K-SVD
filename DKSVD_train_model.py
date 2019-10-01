"""
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import Deep_KSVD
from scipy import linalg

# List of the test image names BSD68:
file_test = open('test_gray.txt', 'r')
onlyfiles_test = []
for e in file_test:
    onlyfiles_test.append(e[:-1])

# List of the train image names:
file_train = open('train_gray.txt', 'r')
onlyfiles_train = []
for e in file_train:
    onlyfiles_train.append(e[:-1])

# Rescaling in [-1, 1]:
mean = 255/2
std = 255/2
data_transform = transforms.Compose([Deep_KSVD.Normalize(mean = mean, std = std),
                                                            Deep_KSVD.ToTensor()])
# Noise level:
sigma = 25
#Sub Image Size:
sub_image_size = 128
# Training Dataset:
my_Data_train = Deep_KSVD.mydataset_sub_images(root_dir =  "gray", image_names = onlyfiles_train,
    sub_image_size = sub_image_size, sigma = sigma, transform = data_transform)
# Test Dataset:
my_Data_test = Deep_KSVD.mydataset_full_images(root_dir =  "gray", image_names = onlyfiles_test,
                sigma = sigma, transform = data_transform)

# Dataloader of the test set:
num_images_test = 5
indices_test = np.random.randint(0,68,num_images_test).tolist()
my_Data_test_sub = torch.utils.data.Subset(my_Data_test, indices_test)
dataloader_test = DataLoader(my_Data_test_sub, batch_size = 1,
                        shuffle = False, num_workers = 0)

# Dataloader of the training set:
batch_size = 1
dataloader_train = DataLoader(my_Data_train, batch_size = batch_size,
                        shuffle = True, num_workers = 0)

#Create a file to see the output during the training:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
file_to_print = open('results_training.csv','w')
file_to_print.write(str(device) + '\n')
file_to_print.flush()

# Initialization:
patch_size = 8
m = 16
Dict_init = Deep_KSVD.Init_DCT(patch_size, m)
Dict_init = Dict_init.to(device)

c_init = linalg.norm(Dict_init,ord=2)**2
c_init = torch.FloatTensor((c_init,))
c_init = c_init.to(device)

w_init = torch.normal(mean = 1, std = 1/10*torch.ones(patch_size**2)).float()
w_init = w_init.to(device)

D_in, H_1, H_2, H_3, D_out_lam, T, min_v, max_v = 64, 128, 64, 32, 1, 5, -1, 1
model  = Deep_KSVD.DenoisingNet_MLP(patch_size, D_in, H_1, H_2, H_3,
                D_out_lam, T, min_v, max_v,Dict_init,c_init,w_init,device)
model.to(device)

# Construct our loss function and an Optimizer:
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

start = time.time()
epochs = 3
running_loss = 0.0
print_every = 1
train_losses, test_losses = [], []
for epoch in range(epochs):  # loop over the dataset multiple times
    for i, (sub_images, sub_images_noise) in enumerate(dataloader_train, 0):
        # get the inputs
        sub_images, sub_images_noise = sub_images.to(device), sub_images_noise.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(sub_images_noise)
        loss = criterion(outputs, sub_images)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % print_every == print_every - 1:    # print every x mini-batches
            train_losses.append(running_loss/print_every)

            end = time.time()
            time_curr = end - start
            file_to_print.write('time:'+' ' + str(time_curr) + '\n')
            start = time.time()

            with torch.no_grad():
                test_loss = 0

                for patches_t, patches_noise_t in dataloader_test:
                    patches, patches_noise = patches_t.to(device), patches_noise_t.to(device)
                    outputs = model(patches_noise)
                    loss = criterion(outputs, patches)
                    test_loss += loss.item()

                test_loss = test_loss/len(dataloader_test)


            end = time.time()
            time_curr = end - start
            file_to_print.write('time:'+' ' + str(time_curr) + '\n')
            start = time.time()

            test_losses.append(test_loss)
            s = '[%d, %d] loss_train: %f, loss_test: %f' % (epoch + 1, (i + 1) * batch_size,
            running_loss / print_every, test_loss)
            s = s + '\n'
            file_to_print.write(s)
            file_to_print.flush()
            running_loss = 0.0

        if i % (10 * print_every) == (10 * print_every)-1:
            torch.save(model.state_dict(), 'model.pth')
            np.savez('losses.npz', train = np.array(test_losses), test = np.array(train_losses))


file_to_print.write('Finished Training')
