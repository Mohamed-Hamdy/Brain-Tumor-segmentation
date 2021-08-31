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
        <li><a href="#Problem Definition">Problem Definition</a></li>
      </ul>
      <ul>
        <li><a href="#Motivation and Objective">Motivation and Objective</a></li>
      </ul>
      <ul>
        <li><a href="#Flow of the Modelivation and Objective">Flow of the Modelivation and Objective</a></li>
      </ul>
    </li>
    <li>
      <a href="#Dataset">Dataset</a>
    </li>
    <li>
      <a href="#Model Phases">Explanation Of The Model</a>
      <ul>
        <li><a href="#Phase1">Preprocessing phase</a></li>
      </ul>
      <ul>
        <li><a href="#phase2">Training Phase</a></li>
      </ul>
      <ul>
        <li><a href="#phase3">Prediction Phase</a></li>
      </ul>
      </li>
    <li>
      <a href="#Result Analysis">Result Analysis</a>
      <ul>
        <li><a href="#segmentation result">Segmentation Result</a></li>
      </ul>
      <ul>
        <li><a href="#classification result">Classification Result</a></li>
      </ul>
      <ul>
        <li><a href="#implementation analysis">Implementation Analysis</a></li>
      </ul>
      <ul>
        <li><a href="#comparative analysis">Comparative Analysis</a></li>
      </ul>
     </li>
  </ol>
</details>




<!-- General C++ Problems -->
## General
Brain tumor segmentation seeks to separate healthy tissue from tumorous regions such as the advancing tumor, necrotic core and surrounding edema. This is an essential step in diagnosis and treatment planning, both of which need to take place quickly in the case of a malignancy in order to maximize the likelihood of successful treatment. Due to the slow and tedious nature of manual segmentation, there is a high demand for computer algorithms that can do this quickly and accurately.
<hr>

### Overview
Brain tumor is one of the most dangerous types of cancer caused by the 5-year survival rate is only about 36%. Accurate diagnosis of the brain the tumor is critical to treatment planning. A major challenge in brain tumor treatment planning and quantitative evaluation is determination of the tumor extent.Manual segmentation of brain tumor extent from 3D MRI volumes is a very time-consuming task and the performance is highly relied on operator’s experience. In this context, a reliable fully automatic segmentation method for the brain tumor segmentation is necessary for an efficient measurement of the tumor extent. In this study, we propose two a fully automatic method for brain tumor segmentation, which is developed using U-Net based model and convent based model deep convolutional networks. Automatic segmentation of brain tumors from medical images is important for clinical assessment and treatment planning of brain tumors.
<hr>

### Problem Definition
One of the reasons for the rise in death rates in today's society is the development of brain tumors. Any mass caused by aberrant or uncontrolled cell development is referred to as a tumor. A benign or malignant brain tumor can exist. The structure of a benign brain tumor is uniform, and it does not contain active (cancer) cells, whereas the structure of a malignant brain tumor is non-uniform (heterogeneous), and it contains active cells. We must first read an MRI image of the brain before applying image segmentation to detect a brain tumor. We demonstrate an effective strategy for removing noise from MRI images and detecting brain malignancies.
	In general, the body consists of many types of cells, and each type has its own functions. The most cells in the body grow and divide in a regular manner to produce more cells. When the body needs it to maintain its health and safety of work. And when you lose your cells, its ability to control and discipline the process of its growth, and the rate of its division increases without any irregularity leading to the formation of extra cells of tissue called a tumor that can be benign or malicious.
<hr>

### Motivation and Objective
In medical imaging of a brain tumor, our major goal is to extract meaningful and trustworthy information from these images with the least amount of error feasible, and then to determine the tumor's area in the image. To assist medical personnel in determining the size and severity of the tumor in order to make the best treatment option. The goal of this research is to create an algorithm that uses convolutional neural networks and segmentation techniques to extract a tumor image from an MRI brain image.
<hr>

### Flow of the Model
<img src="https://github.com/Mohamed-Hamdy/Brain-Tumor-segmentation/blob/master/images/flow%20of%20project.png">
<hr>
