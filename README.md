# Brain-Tumor-segmentation
A deep learning application based on approach for brain tumor MRI segmentation.


<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#General">Introduction</a>
      <ul>
        <li><a href="#Overview">Overview</a></li>
      </ul>
      <ul>
        <li><a href="#Problem_Definition">Problem Definition</a></li>
      </ul>
      <ul>
        <li><a href="#Motivation_and_Objective">Motivation and Objective</a></li>
      </ul>
      <ul>
        <li><a href="#Flow_of_the_Modelivation_and_Objective">Flow of the Modelivation and Objective</a></li>
      </ul>
    </li>
    <li>
      <a href="#Dataset">Dataset</a>
    </li>
    <li>
      <a href="#Model_Phases">Explanation Of The Model</a>
      <ul>
        <li><a href="#Preprocessing_Phase">Preprocessing phase</a></li>
      </ul>
      <ul>
        <li><a href="#Training_phase">Training Phase</a></li>
      </ul>
      <ul>
        <li><a href="#Prediction_Phase">Prediction Phase</a></li>
      </ul>
      </li>
    <li>
      <a href="#Result_Analysis">Result Analysis</a>
      <ul>
        <li><a href="#segmentation_result">Segmentation Result</a></li>
      </ul>
      <ul>
        <li><a href="#classification_result">Model Graphs</a></li>
      </ul>
      <ul>
        <li><a href="#implementation_analysis">Implementation Analysis</a></li>
      </ul>
      <ul>
        <li><a href="#comparative_analysis">Comparative Analysis</a></li>
      </ul>
     </li>  
    <li>
      <a href="#How_to_run">How To Run</a>
     </li>
     <li>
      <a href="#Project_Team_Members">Project Team Members</a>
     </li>
    <li>
      <a href="#Future_work">Future work</a>
     </li>
     <li>
      <a href="https://github.com/Mohamed-Hamdy/Brain-Tumor-segmentation/blob/master/Project%20documentation.pdf">Project Documentation</a>
     </li>
	  
  </ol>
</details>




<!-- General C++ Problems -->
## General
Brain tumor segmentation seeks to separate healthy tissue from tumorous regions such as the advancing tumor, necrotic core and surrounding edema. This is an essential step in diagnosis and treatment planning, both of which need to take place quickly in the case of a malignancy in order to maximize the likelihood of successful treatment. Due to the slow and tedious nature of manual segmentation, there is a high demand for computer algorithms that can do this quickly and accurately.

## Overview
Brain tumor is one of the most dangerous types of cancer caused by the 5-year survival rate is only about 36%. Accurate diagnosis of the brain the tumor is critical to treatment planning. A major challenge in brain tumor treatment planning and quantitative evaluation is determination of the tumor extent.Manual segmentation of brain tumor extent from 3D MRI volumes is a very time-consuming task and the performance is highly relied on operator’s experience. In this context, a reliable fully automatic segmentation method for the brain tumor segmentation is necessary for an efficient measurement of the tumor extent. In this study, we propose two a fully automatic method for brain tumor segmentation, which is developed using U-Net based model and convent based model deep convolutional networks. Automatic segmentation of brain tumors from medical images is important for clinical assessment and treatment planning of brain tumors.

## Problem_Definition
One of the reasons for the rise in death rates in today's society is the development of brain tumors. Any mass caused by aberrant or uncontrolled cell development is referred to as a tumor. A benign or malignant brain tumor can exist. The structure of a benign brain tumor is uniform, and it does not contain active (cancer) cells, whereas the structure of a malignant brain tumor is non-uniform (heterogeneous), and it contains active cells. We must first read an MRI image of the brain before applying image segmentation to detect a brain tumor. We demonstrate an effective strategy for removing noise from MRI images and detecting brain malignancies.
	In general, the body consists of many types of cells, and each type has its own functions. The most cells in the body grow and divide in a regular manner to produce more cells. When the body needs it to maintain its health and safety of work. And when you lose your cells, its ability to control and discipline the process of its growth, and the rate of its division increases without any irregularity leading to the formation of extra cells of tissue called a tumor that can be benign or malicious.

## Motivation_and_Objective
In medical imaging of a brain tumor, our major goal is to extract meaningful and trustworthy information from these images with the least amount of error feasible, and then to determine the tumor's area in the image. To assist medical personnel in determining the size and severity of the tumor in order to make the best treatment option. The goal of this research is to create an algorithm that uses convolutional neural networks and segmentation techniques to extract a tumor image from an MRI brain image.

## Flow_of_the_Model
<img src="https://github.com/Mohamed-Hamdy/Brain-Tumor-segmentation/blob/master/images/flow%20of%20project.png" width="500" height="300" >

## Dataset
Since BraTS'16, data sets utilized for the challenges of this year have been upgraded by professional board-certified neurodiologists, including more routine multimodal 3T MRI images and all ground-truth labelling.
The training, validation and testing data on the BraTS challenge for this year will provide ample multi-institutional clinically acquired pre-operational glioblastoma MRI (GBM/HGG) and lower glioma (LGG) multi-institutional scan with pathologically confirmed diagnosis and available operational system.
The challenge training dataset BraTS 2017 comprises of 210 MRI preoperative scans from HGG subjects and 75 LGG subject images, while the 2018 challenge validation dataset BraTS has 66 different 3D-MRI multimodality scans.
Images from 19 distinct facilities were gathered from various suppliers using the MR scanners and 3T field strength. 
They include co-registered MRI native (T1) and T1-weighted (T1Gd) contrasts as well as MRI co-registered T2 and MRI (FLAIR) attensified fluid (T2). 
All BraTS'2017 3D-MRI data sets are 240 per 240 per 155.
They are distributed, co-registered and interpolated to the same anatomical template (1 mm3). 
All RMI volumes were manually segmented by one to four raters, with expert neuro-radiologists having confirmed their annotations. The edema, necrosis and non-simulating tumors and active/improving tumors each were segmented.

## Explanation_Of_The_Model

<h4>Preprocessing_Phase</h4>
First, a minimal pre-processing of MRI data is applied. The 1% highest and lowest intensities were removed, then each modality of MR images was normalized by subtracting the mean and dividing by the standard deviation of the intensities within the slice. To address the class imbalance problem in the data, data augmentation technique were employed. This consists in adding new synthetic images by performing operations and transformations on data and the corresponding manual tumors segmentation images obtained by human experts (ground truth data). The transformations comprise rotation, translation, and horizontal ﬂipping and mirroring.

<h4>Training_phase</h4>
The CNN used in this study has a similar architecture as that of U-net [1]. Our network architecture can be seen in Fig. 3. It consists of a contracting path (left side) and an expanding path (right side). The contracting path consists of 3 pre-activated residual blocks, as in [2, 3], instead of plain blocks in the original U-net. Each block has two convolution units each of which comprises a Batch Normalization (BN) layer, an activation function, called Parametric Rectiﬁed Linear Unit (PReLU) [4],instead of ReLU function used in the original architecture [1], and a convolutional layer, like in [5], instead of using Maxpooling [1], with Padding = 2, Stride = 1 and a 3 x 3 size ﬁlter. For down sampling, a convolution layer with a 2 x 2 ﬁlter and a stride of 2 is applied. At each down sampling step, the number of feature channels is doubled. The contracting path is followed by a fourth residual unit that acts as a bridge to connect both paths. In the same way, the expanding path is built using 3 residual blocks. Prior to each block, there is an upsampling operation which increases the feature map size by 2, followed by a 2 x 2 convolution and a concatenation with the feature maps corresponding to the contracting path. In the last layer of the expanding path, a 1 x 1 convolution with the Softmax activation function is used to map the multi-channel feature maps to the desired number of classes. In total, the proposed network model contains 7 residual blocks, 25 convolution layers, 15 layers of BN and 10159748 parameters to optimize. The designed network was trained with axial slices extracted from training MRI set, including HGG and LGG cases, and the corresponding ground truth segmentations. The goal is to ﬁnd the network parameters (weights and biases) that minimize a loss function. In this work, this can be achieved by using Stochastic Gradient Descent algorithm (SGD) [6], at each iteration SGD updates the parameters towards the opposite direction of the gradients. In our network model, we used a loss function categorical crossentropy because it’s the loss function that benefit with our dataset.
<h3>Model Architecture</h3>
<img src="https://github.com/Mohamed-Hamdy/Brain-Tumor-segmentation/blob/master/images/model%20Architecture.png">

<h4>Prediction_Phase </h4>
After network training, prediction may be performed. This step consists to provide the network with the four MRI modalities of an unsegmented volume that it has never processed or encountered before, and it must be able to return a segmented image.

## ResultcAnalysis
<h4>Segmentation Result</h4>
<img src="https://github.com/Mohamed-Hamdy/Brain-Tumor-segmentation/blob/master/images/segmentation%20result.png">


## <h4>Model_Graphs</h4>
<img src="https://github.com/Mohamed-Hamdy/Brain-Tumor-segmentation/blob/master/images/Graphs.png">


## <h4>Implementation_Analysis</h4>
This project is very complex and need a very good and power full hardware to get done so we use google Colab as an implementation tool for the two models and the specifications of the tool 
* 25 GB of Ram 
* 100 GB of storage
* Nvidia K80 12 GB as a GPU for training the models 
But google colab don’t fit all the dataset so its fit about 20% only so we are able to find another device more powerfull than google colab and use it to run the project.
<img src="https://github.com/Mohamed-Hamdy/Brain-Tumor-segmentation/blob/master/images/Gui%20Screen.png">

## <h4>Implementation_Analysis</h4>
This project is very complex and need a very good and power full hardware to get done so we use google Colab as an implementation tool for the two models and the specifications of the tool 
* 25 GB of Ram 
* 100 GB of storage
* Nvidia K80 12 GB as a GPU for training the models 
But google colab don’t fit all the dataset so its fit about 20% only so we are able to find another device more powerfull than google colab and use it to run the project.
<img src="https://github.com/Mohamed-Hamdy/Brain-Tumor-segmentation/blob/master/images/Gui%20Screen.png">

## <h4>comparative_analysis</h4>
In this chapter we will represent the final model accuracy and compare our two models with other models that solve the problem of brain tumor detection and the following table show this comparison:
<img src="https://github.com/Mohamed-Hamdy/Brain-Tumor-segmentation/blob/master/images/Table.png">



## <h4>How_to_run</h4>
* Prepare Your Dataset 
* Run The Extract Patches File 
* use the files (Two files store the patches and lables)
* Run the train files to Train model 
* use the files (Two files one for weights of model and another json file)
* Run the Prediction files to get the Result without background of Brain 
* Run the Gui file if you want to get full Result like in <a href="https://github.com/Mohamed-Hamdy/Brain-Tumor-segmentation/blob/master/images/Gui%20Screen.png">this Screenshot</a>

### Project_Team_Members <br>
Mohamed Hamdy Mohamed - [@LinkedIn](https://www.linkedin.com/in/mohamed-hamdy-0155b2173/) - LinkedIn<br>
Mohamed Alaa Kelany - [@LinkedIn](https://www.linkedin.com/in/mohamed-alaa-4368551b2/) - LinkedIn<br>
Mohamed Ali Farouk - [@LinkedIn](https://www.linkedin.com/in/mohamed-ali-b42320185/) - LinkedIn<br>
Ahamed Gomaa Farouk - [@LinkedIn](https://www.linkedin.com/in/ahmad-gomma-3873441aa/) - LinkedIn

## <h4>Future_work</h4>
A more powerful GPU is proposed as future development to speed up CNN's learning phase further. A wider range of CNN topologies and other data increase approaches can therefore be tested. Other fascinating perspectives, for example stacking and blending in tumor core and active tumor areas, are also used to increase segmentation performance with ensemble learning approaches.<br>
More accurate step that can be done in the future is classification which will help the medical staff to know not only the tumor location and region but also the grade of the tumor and the best treatment methods.


