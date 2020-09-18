###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - define the training
#
##################################################
import numpy as np
import configparser as ConfigParser
import os, time

from torch.autograd import Variable
from torchviz import make_dot
import matplotlib.pyplot as plt
import itertools
import tqdm
import pickle
import imageio
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
import sys

sys.path.insert(0, './lib/')
from help_functions import *
from Adaptive_NL2_Block import Adaptive_NL2_Block2D
from PCA_Net_3down_6AM import *
# function to obtain data for training/testing (validation)
from extract_patches import get_data_training
from extract_patches import recompone
from extract_patches import recompone_overlap
from extract_patches import paint_border
from extract_patches import kill_border
from extract_patches import pred_only_FOV
from extract_patches import pred_not_only_FOV
from extract_patches import get_data_testing
from extract_patches import get_data_testing_overlap
# pre_processing.py
from pre_processing import my_PreProc

# ========= Load settings from Config file
config = ConfigParser.RawConfigParser()
config.read('configuration.txt')
# patch to the datasets
path_data = config.get('data paths', 'path_local')
# Experiment name
name_experiment = config.get('experiment name', 'name')
# training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))

# ============ Load the data and divided in patches
patches_imgs_train, patches_masks_train = get_data_training(
    DRIVE_train_imgs_original=path_data + config.get('data paths', 'train_imgs_original'),
    DRIVE_train_groudTruth=path_data + config.get('data paths', 'train_groundTruth'),  # masks
    patch_height=int(config.get('data attributes', 'patch_height')),
    patch_width=int(config.get('data attributes', 'patch_width')),
    N_subimgs=int(config.get('training settings', 'N_subimgs')),
    inside_FOV=config.getboolean('training settings', 'inside_FOV')
    # select the patches only inside the FOV  (default == True)
)

# Save a sample of what you're feeding to the neural network
N_sample = min(patches_imgs_train.shape[0], 40)
visualize(group_images(patches_imgs_train[0:N_sample, :, :, :], 5),
          './' + name_experiment + '/' + "sample_input_imgs")  # .show()
visualize(group_images(patches_masks_train[0:N_sample, :, :, :], 5),
          './' + name_experiment + '/' + "sample_input_masks")  # .show()

# split = 0.9
split = 1.0
N = int(patches_imgs_train.shape[0] * split)
print("%d for training" % (N))
# TensorDataset将数据以tensor形式放入数据库中
train_dataset = torch.utils.data.TensorDataset(torch.Tensor(patches_imgs_train[0:N, :, :, :]),
                                               torch.Tensor(patches_masks_train[0:N, :, :, :]))
# DataLoader将数据库中的数据shuffle，每次从数据库中按batch抛出一批数据
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
'''
val_dataset = torch.utils.data.TensorDataset(torch.Tensor(patches_imgs_train[N:, :, :, :]),
                                             torch.Tensor(patches_masks_train[N:, :, :, :]))
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
'''

# =========== Construct and save the model arcitecture =====
# construct the model
n_ch = patches_imgs_train.shape[1]
patch_height = patches_imgs_train.shape[2]
patch_width = patches_imgs_train.shape[3]

G = PCA_Net()
G.weight_init(mean=0.0, std=0.02)   # 网络参数初始化
# G = nn.DataParallel(G)
G.cuda()

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()
CROSS_loss = nn.CrossEntropyLoss()
# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=0.001, betas=(0.9, 0.999))

# 简单测试
print("Check: final output of the network:")
t = torch.Tensor(1, 1, patch_height, patch_width).cuda()
print((G(t)).shape)

# ========== Save and test the last model ===================

train_hist = {}
train_hist['G_losses'] = []
def show_train_hist(hist, show=False, save=True, path='./' + name_experiment + '/Train_hist.png'):
    x = range(len(hist['G_losses']))
    y1 = hist['G_losses']
    plt.plot(x, y1, label='G_loss')
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(path)
    if show:
        plt.show()
    else:
        plt.close()

# Train Model
F1 = 0
for epoch in range(N_epochs):
    G_losses = []
    epoch_start_time = time.time()

    # train PCA Net
    for idx, (imgs, g_truth) in tqdm.tqdm(enumerate(train_loader)):
        mini_batch = imgs.size()[0]
        imgs, g_truth = Variable(imgs.cuda()), Variable(g_truth.cuda())
        if (idx + 1) % 1 == 0:
            G.zero_grad()
            G_optimizer.zero_grad()
            G_result = G(imgs)

            Seg_Loss = BCE_loss(G_result, g_truth)
            Seg_Loss.backward()

            G_optimizer.step()
            G_loss = Seg_Loss
            G_losses.append(G_loss.item())
    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    print('[%d/%d] - ptime: %.2f, loss_g: %.3f' % (
        (epoch + 1), N_epochs, per_epoch_ptime, torch.mean(torch.FloatTensor(G_losses))))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))

    if (epoch + 1) % 1 == 0:
        show_train_hist(train_hist, save=True, path='./'+name_experiment+'/train_hist_%d.png' % ((epoch + 1)))

    '''
    Test
    '''
    # ========= CONFIG FILE TO READ FROM =======
    config = ConfigParser.RawConfigParser()
    config.read('configuration.txt')
    # ===========================================
    # run the training on invariant or local
    path_data = config.get('data paths', 'path_local')

    # original test images (for FOV selection)
    DRIVE_test_imgs_original = path_data + config.get('data paths', 'test_imgs_original')
    test_imgs_orig = load_hdf5(DRIVE_test_imgs_original)
    full_img_height = test_imgs_orig.shape[2]
    full_img_width = test_imgs_orig.shape[3]
    # the border masks provided by the DRIVE
    DRIVE_test_border_masks = path_data + config.get('data paths', 'test_border_masks')
    test_border_masks = load_hdf5(DRIVE_test_border_masks)
    # dimension of the patches
    patch_height = int(config.get('data attributes', 'patch_height'))
    patch_width = int(config.get('data attributes', 'patch_width'))
    # the stride in case output with average
    stride_height = int(config.get('testing settings', 'stride_height'))
    stride_width = int(config.get('testing settings', 'stride_width'))
    assert (stride_height < patch_height and stride_width < patch_width)
    # model name
    name_experiment = config.get('experiment name', 'name')
    path_experiment = './' + name_experiment + '/'
    # N full images to be predicted
    Imgs_to_test = int(config.get('testing settings', 'full_images_to_test'))
    # Grouping of the predicted images
    N_visual = int(config.get('testing settings', 'N_group_visual'))
    # ====== average mode ===========
    average_mode = config.getboolean('testing settings', 'average_mode')

    # ============ Load the data and divide in patches==============
    patches_imgs_test = None
    new_height = None
    new_width = None
    masks_test = None
    patches_masks_test = None
    if average_mode == True:
        patches_imgs_test, new_height, new_width, masks_test = get_data_testing_overlap(
            DRIVE_test_imgs_original=DRIVE_test_imgs_original,  # original
            DRIVE_test_groudTruth=path_data + config.get('data paths', 'test_groundTruth'),  # masks
            Imgs_to_test=int(config.get('testing settings', 'full_images_to_test')),
            patch_height=patch_height,
            patch_width=patch_width,
            stride_height=stride_height,
            stride_width=stride_width
        )
    else:
        patches_imgs_test, patches_masks_test = get_data_testing(
            DRIVE_test_imgs_original=DRIVE_test_imgs_original,  # original
            DRIVE_test_groudTruth=path_data + config.get('data paths', 'test_groundTruth'),  # masks
            Imgs_to_test=int(config.get('testing settings', 'full_images_to_test')),
            patch_height=patch_height,
            patch_width=patch_width,
        )
    batch_size = 32
    if average_mode == True:
        test_dataset = torch.utils.data.TensorDataset(torch.Tensor(patches_imgs_test), torch.Tensor(patches_imgs_test))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        test_dataset = torch.utils.data.TensorDataset(torch.Tensor(patches_imgs_test), torch.Tensor(patches_masks_test))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ================ Run the prediction of the patches ==================================
    print("================start predict===================")
    best_last = config.get('testing settings', 'best_last')
    predictions = np.empty(patches_imgs_test.shape)
    for idx, (imgs, masks) in tqdm.tqdm(enumerate(test_loader)):
        imgs = imgs.cuda()
        mini_batch = imgs.size()[0]
        outputs = G(imgs)
        predictions[idx * batch_size:idx * batch_size + mini_batch, :, :, :] = outputs.cpu().detach().numpy()

    print ("predicted images size :")
    print (predictions.shape)
    pred_patches = predictions
    sample = predictions[0, :, :, :]
    result = np.sum((sample >= 0.5).astype(float))
    print(result)

    # ========== Elaborate and visualize the predicted images ====================
    pred_imgs = None
    orig_imgs = None
    gtruth_masks = None
    if average_mode == True:
        pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)  # predictions
        orig_imgs = my_PreProc(test_imgs_orig[0:pred_imgs.shape[0], :, :, :])  # originals
        gtruth_masks = masks_test  # ground truth masks
    else:
        pred_imgs = recompone(pred_patches, 13, 12)  # predictions
        orig_imgs = recompone(patches_imgs_test, 13, 12)  # originals
        gtruth_masks = recompone(patches_masks_test, 13, 12)  # masks
    # apply the DRIVE masks on the repdictions #set everything outside the FOV to zero!!
    kill_border(pred_imgs, test_border_masks)  # DRIVE MASK  #only for visualization
    # back to original dimensions
    orig_imgs = orig_imgs[:, :, 0:full_img_height, 0:full_img_width]
    pred_imgs = pred_imgs[:, :, 0:full_img_height, 0:full_img_width]
    gtruth_masks = gtruth_masks[:, :, 0:full_img_height, 0:full_img_width]

    # ====== Evaluate the results
    print("\n\n========  Evaluate the results =======================")
    y_scores, y_true = pred_not_only_FOV(pred_imgs, gtruth_masks)  # returns data not only inside the FOV

    # Confusion matrix
    threshold_confusion = 0.5
    print("\nConfusion matrix:  Custom threshold (for positive) of " + str(threshold_confusion))
    y_pred = np.empty((y_scores.shape[0]))
    for i in range(y_scores.shape[0]):
        if y_scores[i] >= threshold_confusion:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    confusion = confusion_matrix(y_true, y_pred)
    # print (confusion)
    accuracy = 0
    if float(np.sum(confusion)) != 0:
        accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
    print("Global Accuracy: " + str(accuracy))
    specificity = 0
    if float(confusion[0, 0] + confusion[0, 1]) != 0:
        specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
    print("Specificity: " + str(specificity))
    sensitivity = 0
    if float(confusion[1, 1] + confusion[1, 0]) != 0:
        sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
    print("Sensitivity: " + str(sensitivity))
    precision = 0
    if float(confusion[1, 1] + confusion[0, 1]) != 0:
        precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
    print ("Precision: " + str(precision))

    # Jaccard similarity index
    jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)
    # print("\nJaccard similarity score: " + str(jaccard_index))

    # F1 score
    F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
    print ("F1-SCORE: " + str(F1_score))

    if F1_score > F1:
        torch.save(G.state_dict(), "./" + name_experiment + "/PCA_param_FP.pkl")
        F1 = F1_score

show_train_hist(train_hist, save=True, path='./'+name_experiment+'/train_hist.png.png')