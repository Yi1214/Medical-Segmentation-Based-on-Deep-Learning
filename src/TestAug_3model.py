###################################################
#
#   Script to
#   - Calculate prediction of the test dataset
#   - Calculate the parameters to evaluate the prediction
#
##################################################

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import configparser as ConfigParser
from matplotlib import pyplot as plt
import tqdm

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score

import sys
sys.path.insert(0, './lib/')
import  UNet_3down_631_123 as PCA3
import  UNet_3down_321_123 as PCA2
import  UNet_3down_111_124 as PCA1
# help_functions.py
from help_functions import *
# extract_patches.py
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

from PIL import Image
from torchvision import transforms as T


def get_config(config_file):
    config = ConfigParser.RawConfigParser()
    config.read(config_file)
    #===========================================
    # run the training on invariant or local
    path_data = config.get('data paths', 'path_local')

    #original test images (for FOV selection)
    DRIVE_test_imgs_original = path_data + config.get('data paths', 'test_imgs_original')
    test_imgs_orig = load_hdf5(DRIVE_test_imgs_original)
    full_img_height = test_imgs_orig.shape[2]
    full_img_width = test_imgs_orig.shape[3]
    #the border masks provided by the DRIVE
    DRIVE_test_border_masks = path_data + config.get('data paths', 'test_border_masks')
    test_border_masks = load_hdf5(DRIVE_test_border_masks)
    # dimension of the patches
    patch_height = int(config.get('data attributes', 'patch_height'))
    patch_width = int(config.get('data attributes', 'patch_width'))
    #the stride in case output with average
    stride_height = int(config.get('testing settings', 'stride_height'))
    stride_width = int(config.get('testing settings', 'stride_width'))
    assert (stride_height < patch_height and stride_width < patch_width)
    #model name
    name_experiment = config.get('experiment name', 'name')
    path_experiment = './' +name_experiment +'/'
    #path_experiment = './' +'testloss2' +'/'
    #N full images to be predicted
    Imgs_to_test = int(config.get('testing settings', 'full_images_to_test'))
    #Grouping of the predicted images
    N_visual = int(config.get('testing settings', 'N_group_visual'))
    return config, path_data, DRIVE_test_imgs_original, test_imgs_orig, full_img_height, full_img_width, DRIVE_test_border_masks, test_border_masks, patch_height, patch_width, stride_height, stride_width, name_experiment, path_experiment, Imgs_to_test, N_visual

# ================ Get test patches ==================================
def get_test_data(DRIVE_test_imgs_original,
                  path_data,
                  config,
                  patch_height,
                  patch_width,
                  stride_height,
                  stride_width):
    # new_height, new_width 为DRIVE_test_imgs_original的height和width
    patches_imgs_test, new_height, new_width, masks_test = get_data_testing_overlap(
        DRIVE_test_imgs_original=DRIVE_test_imgs_original,  # original
        DRIVE_test_groudTruth=path_data + config.get('data paths', 'test_groundTruth'),  # masks
        Imgs_to_test=int(config.get('testing settings', 'full_images_to_test')),
        patch_height=patch_height,
        patch_width=patch_width,
        stride_height=stride_height,
        stride_width=stride_width
    )

    test_dataset = torch.utils.data.TensorDataset(torch.Tensor(patches_imgs_test), torch.Tensor(patches_imgs_test))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return patches_imgs_test, new_height, new_width, masks_test, test_dataset, test_loader

# ================ Run the prediction of the patches ==================================
def predict_test(config, PCA_Net, model_name, test_loader,
                 patches_imgs_test, new_height, new_width, stride_height,
                 stride_width, full_img_height, full_img_width,
                 angle):
    best_last = config.get('testing settings', 'best_last')
    # Load the saved model
    model = PCA_Net
    model.load_state_dict(torch.load(path_experiment + model_name, map_location='cpu'))
    model.cuda()

    # Predict test data
    predictions = np.empty(patches_imgs_test.shape)
    for idx, (imgs, masks) in tqdm.tqdm(enumerate(test_loader)):
        imgs = imgs.cuda()
        mini_batch = imgs.size()[0]
        outputs = model(imgs)
        predictions[idx * batch_size:idx * batch_size + mini_batch, :, :, :] = outputs.cpu().detach().numpy()
    print("predicted images size :")
    print(predictions.shape)
    pred_patches = predictions
    # new_height, new_width 为DRIVE_test_imgs_original的height和width
    pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)  # predictions

    print("pred imgs shape: " + str(pred_imgs.shape))
    ## back to original dimensions
    pred_imgs = pred_imgs[:,:,0:full_img_height,0:full_img_width]
    print("pred imgs shape: " +str(pred_imgs.shape))    # [20,1,584,584]

    print("rotate angle:" + str(angle*90))
    pred_imgs = np.rot90(pred_imgs, angle, (2, 3))       # np.rot90(m,k,(x,y))
    pred_imgs_back = pred_imgs[:,:,0:584,0:565]
    print("pred imgs shape: " + str(pred_imgs_back.shape))
    return pred_imgs_back

# =========== CONFIG FILE TO READ FROM 原图===============
config, path_data, DRIVE_test_imgs_original, test_imgs_orig, full_img_height, full_img_width, \
DRIVE_test_border_masks, test_border_masks, patch_height, patch_width, stride_height, stride_width, \
name_experiment, path_experiment, Imgs_to_test, N_visual \
    = get_config("/data/zhangkai/APA_Test_Augment_padding_DRIVE/PCA-Net_DRIVE_v1/configuration.txt")

config0, path_data0, DRIVE_test_imgs_original0, test_imgs_orig0, full_img_height0, full_img_width0, \
DRIVE_test_border_masks0, test_border_masks0, patch_height0, patch_width0, stride_height0, stride_width0, \
name_experiment0, path_experiment0, Imgs_to_test0, N_visual0 \
    = get_config("/data/zhangkai/APA_Test_Augment_padding_DRIVE/PCA-Net_DRIVE_v1/configuration0.txt")

config1, path_data1, DRIVE_test_imgs_original1, test_imgs_orig1, full_img_height1, full_img_width1, \
DRIVE_test_border_masks1, test_border_masks1, patch_height1, patch_width1, stride_height1, stride_width1, \
name_experiment1, path_experiment1, Imgs_to_test1, N_visual1 \
    = get_config("/data/zhangkai/APA_Test_Augment_padding_DRIVE/PCA-Net_DRIVE_v1/configuration1.txt")

config2, path_data2, DRIVE_test_imgs_original2, test_imgs_orig2, full_img_height2, full_img_width2, \
DRIVE_test_border_masks2, test_border_masks2, patch_height2, patch_width2, stride_height2, stride_width2, \
name_experiment2, path_experiment2, Imgs_to_test2, N_visual2 \
    = get_config("/data/zhangkai/APA_Test_Augment_padding_DRIVE/PCA-Net_DRIVE_v1/configuration2.txt")

config3, path_data3, DRIVE_test_imgs_original3, test_imgs_orig3, full_img_height3, full_img_width3, \
DRIVE_test_border_masks3, test_border_masks3, patch_height3, patch_width3, stride_height3, stride_width3, \
name_experiment3, path_experiment3, Imgs_to_test3, N_visual3 \
    = get_config("/data/zhangkai/APA_Test_Augment_padding_DRIVE/PCA-Net_DRIVE_v1/configuration3.txt")

######## the model to predict#####################

PCA_Net1 = PCA1.PCA_Net()
model_name1 = "PCA_111_124.pkl"

PCA_Net2 = PCA2.PCA_Net()
model_name2 = "PCA_321_123.pkl"

PCA_Net3 = PCA3.PCA_Net()
model_name3 = "PCA_631_123.pkl"

#============ Load the data and divide in patches==============
patches_imgs_test = None
new_height = None
new_width = None
masks_test  = None
patches_masks_test = None
batch_size = 32

patches_imgs_test, new_height, \
new_width, masks_test, test_dataset,\
test_loader = get_test_data(DRIVE_test_imgs_original, path_data, config,
                            patch_height, patch_width, stride_height, stride_width)

patches_imgs_test0, new_height0, \
new_width0, masks_test0, test_dataset0,\
test_loader0 = get_test_data(DRIVE_test_imgs_original0, path_data0, config0,
                            patch_height0, patch_width0, stride_height0, stride_width0)

patches_imgs_test1, new_height1, \
new_width1, masks_test1, test_dataset1,\
test_loader1 = get_test_data(DRIVE_test_imgs_original1, path_data1, config1,
                            patch_height1, patch_width1, stride_height1, stride_width1)

patches_imgs_test2, new_height2, \
new_width2, masks_test2, test_dataset2,\
test_loader2 = get_test_data(DRIVE_test_imgs_original2, path_data2, config2,
                            patch_height2, patch_width2, stride_height2, stride_width2)

patches_imgs_test3, new_height3, \
new_width3, masks_test3, test_dataset3,\
test_loader3 = get_test_data(DRIVE_test_imgs_original3, path_data3, config3,
                            patch_height3, patch_width3, stride_height3, stride_width3)

# =================get predictions and transform to  same size================

pred_imgs = predict_test(config0, PCA_Net1, model_name1, test_loader0,
                         patches_imgs_test0, new_height0, new_width0, stride_height0,
                         stride_width0, full_img_height0, full_img_width0,
                         angle=0)

pred_imgs1 = predict_test(config1, PCA_Net1, model_name1, test_loader1,
                          patches_imgs_test1, new_height1, new_width1, stride_height1,
                          stride_width1, full_img_height1, full_img_width1,
                          angle=3)

pred_imgs2 = predict_test(config2, PCA_Net1, model_name1, test_loader2,
                          patches_imgs_test2, new_height2, new_width2, stride_height2,
                          stride_width2, full_img_height2, full_img_width2,
                          angle=2)

pred_imgs3 = predict_test(config3, PCA_Net1, model_name1, test_loader3,
                          patches_imgs_test3, new_height3, new_width3, stride_height3,
                          stride_width3, full_img_height3, full_img_width3,
                          angle=1)

pred_imgs_ = predict_test(config0, PCA_Net2, model_name2, test_loader0,
                          patches_imgs_test0, new_height0, new_width0, stride_height0,
                          stride_width0, full_img_height0, full_img_width0,
                          angle=0)

pred_imgs1_1 = predict_test(config1, PCA_Net2, model_name2, test_loader1,
                            patches_imgs_test1, new_height1, new_width1, stride_height1,
                            stride_width1, full_img_height1, full_img_width1,
                            angle=3)

pred_imgs2_1 = predict_test(config2, PCA_Net2, model_name2, test_loader2,
                            patches_imgs_test2, new_height2, new_width2, stride_height2,
                            stride_width2, full_img_height2, full_img_width2,
                            angle=2)

pred_imgs3_1 = predict_test(config3, PCA_Net2, model_name2, test_loader3,
                            patches_imgs_test3, new_height3, new_width3, stride_height3,
                            stride_width3, full_img_height3, full_img_width3,
                            angle=1)

pred_imgs_3 = predict_test(config0, PCA_Net3, model_name3, test_loader0,
                           patches_imgs_test0, new_height0, new_width0, stride_height0,
                           stride_width0, full_img_height0, full_img_width0,
                           angle=0)

pred_imgs1_3 = predict_test(config1, PCA_Net3, model_name3, test_loader1,
                            patches_imgs_test1, new_height1, new_width1, stride_height1,
                            stride_width1, full_img_height1, full_img_width1,
                            angle=3)

pred_imgs2_3 = predict_test(config2, PCA_Net3, model_name3, test_loader2,
                            patches_imgs_test2, new_height2, new_width2, stride_height2,
                            stride_width2, full_img_height2, full_img_width2,
                            angle=2)

pred_imgs3_3 = predict_test(config3, PCA_Net3, model_name3, test_loader3,
                            patches_imgs_test3, new_height3, new_width3, stride_height3,
                            stride_width3, full_img_height3, full_img_width3,
                            angle=1)


pred_imgs_total = (pred_imgs + pred_imgs1 + pred_imgs2 + pred_imgs3 + pred_imgs_ + pred_imgs1_1 + pred_imgs2_1 + pred_imgs3_1 + pred_imgs_3 + pred_imgs1_3 + pred_imgs2_3 + pred_imgs3_3)/12
kill_border(pred_imgs_total, test_border_masks)
print("pred_imgs_total transform:")
print(pred_imgs_total.shape)



# ========== Elaborate and visualize the predicted images ====================

orig_imgs = None
gtruth_masks = None

orig_imgs = my_PreProc(test_imgs_orig[0:pred_imgs.shape[0],:,:,:])    # originals
gtruth_masks = masks_test  # ground truth masks


# back to original dimensions
orig_imgs = orig_imgs[:,:,0:full_img_height,0:full_img_width]
gtruth_masks = gtruth_masks[:,:,0:full_img_height,0:full_img_width]
print("Orig imgs shape: " +str(orig_imgs.shape))
print("pred imgs shape: " +str(pred_imgs.shape))
print("Gtruth imgs shape: " +str(gtruth_masks.shape))


# ==============evaluate=================
print("\n\n========  Evaluate the results =======================")
#predictions only inside the FOV
#y_scores, y_true = pred_only_FOV(pred_imgs,gtruth_masks, test_border_masks)  #returns data only inside the FOV
y_scores, y_true = pred_not_only_FOV(pred_imgs_total,gtruth_masks)  #returns data not only inside the FOV
print("Calculating results only inside the FOV:")
print("y scores pixels: " +str(y_scores.shape[0]) +" (radius 270: 270*270*3.14==228906), including background around retina: " +str(pred_imgs.shape[0]*pred_imgs.shape[2]*pred_imgs.shape[3]) +" (584*565==329960)")
print("y true pixels: " +str(y_true.shape[0]) +" (radius 270: 270*270*3.14==228906), including background around retina: " +str(gtruth_masks.shape[2]*gtruth_masks.shape[3]*gtruth_masks.shape[0])+" (584*565==329960)")
print(y_true)
y_true = y_true.astype(int)
print(y_true)
#Area under the ROC curve
#fpr, tpr, thresholds = roc_curve((y_true), y_scores)
fpr, tpr, thresholds = roc_curve((y_true), y_scores)
AUC_ROC = roc_auc_score(y_true, y_scores)
# test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
print("\nArea under the ROC curve: " +str(AUC_ROC))
roc_curve = plt.figure()
plt.plot(fpr,tpr,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
plt.title('ROC curve')
plt.xlabel("FPR (False Positive Rate)")
plt.ylabel("TPR (True Positive Rate)")
plt.legend(loc="lower right")
plt.savefig(path_experiment+"ROC.png")

#Precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
precision = np.fliplr([precision])[0]  #so the array is increasing (you won't get negative AUC)
recall = np.fliplr([recall])[0]  #so the array is increasing (you won't get negative AUC)
AUC_prec_rec = np.trapz(precision,recall)
print("\nArea under Precision-Recall curve: " +str(AUC_prec_rec))
prec_rec_curve = plt.figure()
plt.plot(recall,precision,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
plt.title('Precision - Recall curve')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower right")
plt.savefig(path_experiment+"Precision_recall.png")

#Confusion matrix
threshold_confusion = 0.5
print("\nConfusion matrix:  Custom threshold (for positive) of " +str(threshold_confusion))
y_pred = np.empty((y_scores.shape[0]))
for i in range(y_scores.shape[0]):
    if y_scores[i]>=threshold_confusion:
        y_pred[i]=1
    else:
        y_pred[i]=0
confusion = confusion_matrix(y_true, y_pred)
print (confusion)
accuracy = 0
if float(np.sum(confusion))!=0:
    accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
print("Global Accuracy: " +str(accuracy))
specificity = 0
if float(confusion[0,0]+confusion[0,1])!=0:
    specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
print("Specificity: " +str(specificity))
sensitivity = 0
if float(confusion[1,1]+confusion[1,0])!=0:
    sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
print("Sensitivity: " +str(sensitivity))
precision = 0
if float(confusion[1,1]+confusion[0,1])!=0:
    precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
print ("Precision: " +str(precision))

#Jaccard similarity index
jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)
print("\nJaccard similarity score: " +str(jaccard_index))

#F1 score
F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
print("\nF1 score (F-measure): " +str(F1_score))

#Save the results
file_perf = open(path_experiment+'performances.txt', 'w')
file_perf.write("Area under the ROC curve: "+str(AUC_ROC)
                + "\nArea under Precision-Recall curve: " +str(AUC_prec_rec)
                + "\nJaccard similarity score: " +str(jaccard_index)
                + "\nF1 score (F-measure): " +str(F1_score)
                +"\n\nConfusion matrix:"
                +str(confusion)
                +"\nACCURACY: " +str(accuracy)
                +"\nSENSITIVITY: " +str(sensitivity)
                +"\nSPECIFICITY: " +str(specificity)
                +"\nPRECISION: " +str(precision)
                )
file_perf.close()

pred_imgs = pred_to_imgs3(pred_imgs_total)
visualize(group_images(orig_imgs,N_visual),path_experiment+"all_originals")    # .show()
visualize(group_images(pred_imgs,N_visual),path_experiment+"all_predictions")   # .show()
visualize(group_images(gtruth_masks,N_visual),path_experiment+"all_groundTruths")  # .show()
# visualize results comparing mask and prediction:
assert (orig_imgs.shape[0]==pred_imgs.shape[0] and orig_imgs.shape[0]==gtruth_masks.shape[0])
N_predicted = orig_imgs.shape[0]
group = N_visual
assert (N_predicted%group==0)
for i in range(int(N_predicted/group)):
    orig_stripe = group_images(orig_imgs[i*group:(i*group)+group,:,:,:],group)
    masks_stripe = group_images(gtruth_masks[i*group:(i*group)+group,:,:,:],group)
    pred_stripe = group_images(pred_imgs[i*group:(i*group)+group,:,:,:],group)
    total_img = np.concatenate((orig_stripe,masks_stripe,pred_stripe),axis=0)
    visualize(total_img,path_experiment+name_experiment +"_Original_GroundTruth_Prediction"+str(i))    # .show()



