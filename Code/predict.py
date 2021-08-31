import matplotlib
import random
from glob import glob
import os
import shutil
import SimpleITK as sitk
from evaluation_metrics import *
from model import Unet_model
from PIL import Image, ImageDraw
from sklearn.metrics import confusion_matrix
import json
import pandas as pd
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix
import imageio
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as FunAnimation
from matplotlib.animation import PillowWriter
import nibabel as nib
from colour import Color
from PIL import Image

class Prediction(object):

    def __init__(self, batch_size_test, load_model_path):

        self.batch_size_test = batch_size_test

        unet = Unet_model(img_shape=(240, 240, 4), load_model_weights=load_model_path)
        self.model = unet.model
        print('U-net CNN compiled!\n')

    def predict_volume(self, filepath_image, show):

        '''
        segment the input volume
        INPUT   (1) str 'filepath_image': filepath of the volume to predict
                (2) bool 'show': True to ,
        OUTPUt  (1) np array of the predicted volume
                (2) np array of the corresping ground truth
        '''

        # read the volume
        flair = glob(filepath_image + '/*_flair.nii.gz')
        t2 = glob(filepath_image + '/*_t2.nii.gz')
        gt = glob(filepath_image + '/*_seg.nii.gz')
        t1s = glob(filepath_image + '/*_t1.nii.gz')
        t1c = glob(filepath_image + '/*_t1ce.nii.gz')

        t1 = [scan for scan in t1s if scan not in t1c]
        if (len(flair) + len(t2) + len(gt) + len(t1) + len(t1c)) < 5:
            print("there is a problem here!!! the problem lies in this patient :")

        scans_test = [flair[0], t1[0], t1c[0], t2[0], gt[0]]

        # print(flair[0])
        test_im = [sitk.GetArrayFromImage(sitk.ReadImage(scans_test[i])) for i in range(len(scans_test))]

        f = np.array(test_im[0])
        t1 = np.array(test_im[1])
        t1c = np.array(test_im[2])
        t2 = np.array(test_im[3])
        Gui_predict_images = np.array(test_im[0])

        if os.path.exists('./Result'):
            shutil.rmtree('./Result')
        os.makedirs('./Result')

        if os.path.exists('./Result/flair_axial'):
            shutil.rmtree('./Result/flair_axial')
        os.makedirs('./Result/flair_axial')

        if os.path.exists('./Result/final_axial_prediction_result'):
            shutil.rmtree('./Result/final_axial_prediction_result')
        os.makedirs('./Result/final_axial_prediction_result')

        if os.path.exists('./Result/final_axial_gt_result'):
            shutil.rmtree('./Result/final_axial_gt_result')
        os.makedirs('./Result/final_axial_gt_result')



        fig = plt.figure(frameon=False)
        # plot flair_axial
        for i in range(f.shape[0]):
            plt.xticks([])
            plt.yticks([])
            im = plt.imshow(f[i, :, :], cmap="gray", interpolation='none')
            plt.axis('off')
            path = "Result/flair_axial/" + str(i) + ".png"
            fig.savefig(path, bbox_inches='tight', pad_inches=0.0, dpi=65)
            # plt.show()
        print("Done plotting flair_axial")


        '''
        dir = []
        t1_dir = []
        t2_dir = []
        t1c_dir = []
        for i in range(155):
            png_dir = 'Result/flair/' + str(i) + '.png'
            t1_png_dir = './t1/' + str(i) + '.png'
            t2_png_dir = './t2/' + str(i) + '.png'
            t1c_png_dir = './t1c/' + str(i) + '.png'
            dir.append(png_dir)
            t1_dir.append(t1_png_dir)
            t2_dir.append(t2_png_dir)
            t1c_dir.append(t1c_png_dir)
        images = []
        t1_images = []
        t1c_images = []
        t2_images = []

        for i in range(len(dir)):

            images.append(imageio.imread(dir[i]))
            t1_images.append(imageio.imread(t1_dir[i]))
            t1c_images.append(imageio.imread(t1c_dir[i]))
            t2_images.append(imageio.imread(t2_dir[i]))
        imageio.mimsave('./Gif/flair.gif', images)
        imageio.mimsave('./Gif/t1.gif', t1_images)
        imageio.mimsave('./Gif/t1c.gif', t1c_images)
        imageio.mimsave('./Gif/t2.gif', t2_images)
        '''
        test_im = np.array(test_im).astype(np.float32)
        test_image = test_im[0:4]
        flair_img = test_im[0]
        gt = test_im[-1]
        gt[gt == 4] = 3
        flair_img[flair_img == 4] = 3

        # normalize each slice following the same scheme used for training
        test_image = self.norm_slices(test_image)

        # transform the data to channels_last keras format
        test_image = test_image.swapaxes(0, 1)
        test_image = np.transpose(test_image, (0, 2, 3, 1))

        test_image = test_image[0:155, :, :]  # sub image
        gt = gt[0:155, :, :]
        flair_img = flair_img[0:155, :, :]
        if show:
            verbose = 1

        else:
            verbose = 0
        # predict classes of each pixel based on the model
        prediction = self.model.predict(test_image, batch_size=self.batch_size_test, verbose=verbose)
        prediction = np.argmax(prediction, axis=-1)
        prediction = prediction.astype(np.uint8)

        # reconstruct the initial target values .i.e. 0,1,2,4 for prediction and ground truth
        prediction[prediction == 3] = 4
        gt[gt == 3] = 4
        flair_img[flair_img == 3] = 4

        return np.array(prediction), np.array(gt), np.array(flair_img), Gui_predict_images

    def evaluate_segmented_volume(self, filepath_image, save, show, save_path):
        '''
        computes the evaluation metrics on the segmented volume
        INPUT   (1) str 'filepath_image': filepath to test image for segmentation, including file extension
                (2) bool 'save': whether to save to disk or not
                (3) bool 'show': If true, prints the evaluation metrics
        OUTPUT np array of all evaluation metrics
        '''

        predicted_images, gt, flair_images, Gui_predict_images = self.predict_volume(filepath_image, show)
        gt = gt.astype('uint8')

        predicted_images[predicted_images == 1] = 60
        predicted_images[predicted_images == 2] = 180
        predicted_images[predicted_images == 4] = 120
        gt[gt == 1] = 60
        gt[gt == 2] = 180
        gt[gt == 4] = 120

        axial_gt_frames = []
        axial_predict_frames = []

        axial_prediction_dir = []
        axial_gt_dir = []

        axial_flair_dir = []

        if os.path.exists('./Result/prediction_axial'):
            shutil.rmtree('./Result/prediction_axial')
        os.makedirs('./Result/prediction_axial')

        if os.path.exists('./Result/GT_axial'):
            shutil.rmtree('./Result/GT_axial')
        os.makedirs('./Result/GT_axial')

        for i in range(gt.shape[0]):  # range(155):
            axial_img1 = Image.fromarray(predicted_images[i, :, :], 'L')
            axial_img2 = Image.fromarray(gt[i, :, :], 'L')

            axial_gt_frames.append(axial_img2)
            axial_predict_frames.append(axial_img1)

            axial_img1.save("Result/prediction_axial/" + str(i) + ".png")
            axial_img2.save("Result/GT_axial/" + str(i) + ".png")

            axial_prediction_png_dir = 'Result/prediction_axial/' + str(i) + '.png'
            axial_gt_png_dir = 'Result/GT_axial/' + str(i) + '.png'

            axial_flair_png_dir = 'Result/flair_axial/' + str(i) + '.png'

            axial_prediction_dir.append(axial_prediction_png_dir)
            axial_gt_dir.append(axial_gt_png_dir)
            axial_flair_dir.append(axial_flair_png_dir)

        final_axial_prediction_images_gif = []
        final_axial_gt_images_gif = []
        for k in range(len(axial_prediction_dir)):
            final_prediction_img = cv2.imread(str(axial_prediction_dir[k]), 0)
            final_flair_img = cv2.imread(str(axial_flair_dir[k]), 0)
            for i in range(final_prediction_img.shape[0]):
                for j in range(final_prediction_img.shape[1]):
                    if final_prediction_img[i][j] == 180:
                        final_flair_img[i][j] = 180
                    if final_prediction_img[i][j] == 60:
                        final_flair_img[i][j] = 60
                    if final_prediction_img[i][j] == 120:
                        final_flair_img[i][j] = 120
            f_dir = 'Result/final_axial_prediction_result/' + str(k) + '.png'
            cv2.imwrite(f_dir, final_flair_img)
            final_axial_prediction_images_gif.append(imageio.imread(f_dir))
            predicted_images[k, :, :] = final_axial_prediction_images_gif[k]
        #imageio.mimsave('./Gif/final_axial_prediction.gif', final_axial_prediction_images_gif)
        print("Done axial predict.")

        for k in range(len(axial_gt_dir)):
            final_gt_img = cv2.imread(str(axial_gt_dir[k]), 0)
            final_flair_gt_img = cv2.imread(str(axial_flair_dir[k]), 0)
            for i in range(final_gt_img.shape[0]):
                for j in range(final_flair_gt_img.shape[1]):
                    if final_gt_img[i][j] == 180:
                        final_flair_gt_img[i][j] = 180
                    if final_gt_img[i][j] == 60:
                        final_flair_gt_img[i][j] = 60
                    if final_gt_img[i][j] == 120:
                        final_flair_gt_img[i][j] = 120
            f_gt_dir = 'Result/final_axial_gt_result/' + str(k) + '.png'
            cv2.imwrite(f_gt_dir, final_flair_gt_img)
            final_axial_gt_images_gif.append(imageio.imread(f_gt_dir))
        #imageio.mimsave('./Gif/final_axial_gt.gif', final_axial_gt_images_gif)

        if save:
            tmp = sitk.GetImageFromArray(predicted_images)
            tmp1 = sitk.GetImageFromArray(gt)

            sitk.WriteImage(tmp, 'Result/seg.nii.gz'.format(save_path))
            sitk.WriteImage(tmp, 'Result/{}_p.nii.gz'.format(save_path))
            sitk.WriteImage(tmp1, 'Result/{}_g.nii.gz'.format(save_path))

        # compute the evaluation metrics
        Dice_complete = DSC_whole(predicted_images, gt)
        Dice_enhancing = DSC_en(predicted_images, gt)
        Dice_core = DSC_core(predicted_images, gt)

        Sensitivity_whole = sensitivity_whole(predicted_images, gt)
        Sensitivity_en = sensitivity_en(predicted_images, gt)
        Sensitivity_core = sensitivity_core(predicted_images, gt)

        Specificity_whole = specificity_whole(predicted_images, gt)
        Specificity_en = specificity_en(predicted_images, gt)
        Specificity_core = specificity_core(predicted_images, gt)

        Hausdorff_whole = hausdorff_whole(predicted_images, gt)
        Hausdorff_en = hausdorff_en(predicted_images, gt)
        Hausdorff_core = hausdorff_core(predicted_images, gt)

        if show:
            print("************************************************************")
            print("Dice complete tumor score : {:0.4f}".format(Dice_complete))
            print("Dice core tumor score (tt sauf vert): {:0.4f}".format(Dice_core))
            print("Dice enhancing tumor score (jaune):{:0.4f} ".format(Dice_enhancing))
            print("**********************************************")
            print("Sensitivity complete tumor score : {:0.4f}".format(Sensitivity_whole))
            print("Sensitivity core tumor score (tt sauf vert): {:0.4f}".format(Sensitivity_core))
            print("Sensitivity enhancing tumor score (jaune):{:0.4f} ".format(Sensitivity_en))
            print("***********************************************")
            print("Specificity complete tumor score : {:0.4f}".format(Specificity_whole))
            print("Specificity core tumor score (tt sauf vert): {:0.4f}".format(Specificity_core))
            print("Specificity enhancing tumor score (jaune):{:0.4f} ".format(Specificity_en))
            print("***********************************************")
            print("Hausdorff complete tumor score : {:0.4f}".format(Hausdorff_whole))
            print("Hausdorff core tumor score (tt sauf vert): {:0.4f}".format(Hausdorff_core))
            print("Hausdorff enhancing tumor score (jaune):{:0.4f} ".format(Hausdorff_en))
            print("***************************************************************\n\n")
        return np.array(gt), np.array(predicted_images), np.array((Dice_complete, Dice_core, Dice_enhancing,
                                                                   Sensitivity_whole,
                                                                   Sensitivity_core,
                                                                   Sensitivity_en,
                                                                   Specificity_whole,
                                                                   Specificity_core,
                                                                   Specificity_en,
                                                                   Hausdorff_whole,
                                                                   Hausdorff_core,
                                                                   Hausdorff_en))  # ))

    def predict_multiple_volumes(self, filepath_volumes, save, show):

        gt_array, prediction_array, results = [], [], []
        counter = 0

        for patient in filepath_volumes:
            counter += 1
            tmp1 = patient.split('/')
            # print(counter)
            # print(tmp1[-2])
            # print(tmp1[-1])
            print("\n\nVolume ID: ", tmp1[-2] + '/' + tmp1[-1])
            gt, predicted_images, tmp = self.evaluate_segmented_volume(patient, save=save, show=show,
                                                                       save_path=os.path.basename(patient))
            # save the results of each volume
            results.append(tmp)
            gt_array.append(gt)
            prediction_array.append(predicted_images)
            # save each ID for later use

        res = np.array(results)
        print("mean : ", np.mean(res, axis=0))
        print("std : ", np.std(res, axis=0))
        print("median : ", np.median(res, axis=0))
        print("25 quantile : ", np.percentile(res, 25, axis=0))
        print("75 quantile : ", np.percentile(res, 75, axis=0))
        print("max : ", np.max(res, axis=0))
        print("min : ", np.min(res, axis=0))

        np.savetxt('./Results.out', res)
        # np.savetxt('./Volumes_ID.out', Ids, fmt='%s')

    def norm_slices(self, slice_not):
        '''
            normalizes each slice, excluding gt
            subtracts mean and div by std dev for each slice
            clips top and bottom one percent of pixel intensities
        '''
        normed_slices = np.zeros((4, 155, 240, 240))
        for slice_ix in range(4):
            normed_slices[slice_ix] = slice_not[slice_ix]
            for mode_ix in range(155):
                normed_slices[slice_ix][mode_ix] = self._normalize(slice_not[slice_ix][mode_ix])

        return normed_slices

    def _normalize(self, slice):

        b = np.percentile(slice, 99)
        t = np.percentile(slice, 1)
        slice = np.clip(slice, t, b)
        image_nonzero = slice[np.nonzero(slice)]

        if np.std(slice) == 0 or np.std(image_nonzero) == 0:
            return slice
        else:
            tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
            tmp[tmp == tmp.min()] = -9
            return tmp


if __name__ == "__main__":
    # set arguments
    model_to_load = "./All_dataset_Training_Model.hdf5"
    print("1")
    # paths for the testing data
    path_HGG = glob('E:/GP/Dataset/BRATS2017/Brats17TrainingData/LGG/Brats17_TCIA_639_1')
    # path_LGG = glob('/content/drive/My Drive/Project/BRATS2017/Brats17TrainingData/LGG/**')
    print("2")

    test_path = path_HGG

    np.random.seed(2022)
    np.random.shuffle(test_path)

    # compile the model
    brain_seg_pred = Prediction(batch_size_test=2, load_model_path=model_to_load)

    # predicts each volume and save the results in np array
    brain_seg_pred.predict_multiple_volumes(path_HGG, save=True, show=False)



