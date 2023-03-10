# Diagnosis of breast tumors through Mammography Image Analysis using deep learning algorithms
## Abstract
Breast cancer is now commonly recognised as the second-leading cause of death among women. Computer Aided Detection systems provide significant solutions in the early detection and diagnosis of this cancer. The main objective of this study is to develop a system using deep learning algorithms to classify breast lesions into malignant, benign, and normal. This study contains three phases, image preprocessing phase, applying deep learning models, and evaluating our results base using 10-foldcross validation. Five different scenarios have been designed to evaluate different preprocessing techniques and check them with using two different architectures of Convolutional Neural Networks.
<br><br>The proposed system offers good classification rates. These techniques were applied on the MIAS dataset. Combining image contrast enhancement with class weight technique applied on a 4-layer convolutional neural network showed great performance and promising outcomes using 10-fold cross validation. For the best method, the accuracy, f-measure, recall, and precision reached **80.99%**, **81.07%**, **80.99%**, and **80.99%** respectively. <br>According to the findings of this study, CNNs offer a lot of potential in the field of intelligent medical image diagnosis.<br> 

## Codes
In this work, the most important steps were the preprocessing levels.
Five different scenarios have been designed to evaluate different preprocessing techniques and check them with using two different architectures of Convolutional Neural Networks.
<br>At first, images have been read. Afterwards, class weight technique and rotation augmentation have been applied in order to make our classes balanced. In the last two scenarios image, contrast enhancement with rotation augmentation and class weight technique have been combined.
<br><br> The full report paper of this work named as Breast Tumor Classifying Project and all the codes for preprocessing are in the Preprocessing file.

