# Generation of Diverse Image Data Using Style Transfer for Debiasing Melanoma Classification Models

[Devpost with General Overview, Final Report and Poster](https://devpost.com/software/dl-on-the-dl?ref_content=my-projects-tab&ref_feature=my_projects)

# Data
The data that we used to generate the trained CNN can be found in the ISIC_data folder, which contains 9 different classes of skin cancer. The data used was found from Kaggle (https://www.kaggle.com/c/siim-isic-melanoma-classification) and had the following classifications:
- actinic keratosis
- basal cell carcinoma
- dermatofibroma
- melanoma
- nevus
- pigmented benign keratosis
- seborrheic keratosis
- squamous cell carcinoma
- vascular lesion

# Run the program 
To train the CNN, run the script `python3 cnn_train.py` in the terminal from the code directory. This will save weights of the final trained CNN with the best weights. The best performace for the CNN had a training accuracy of 44%. 

To run the style transfer code, run the script `python3 style_transfer.py` in the terminal from the code directory. Within the scripte, you can change the file path to different melanoma features and skin styles. It will save before and after images. It sucessfully decreased loss between features and styles. 

# References 
The CNN architecture was influenced by VGG16. 

The style transfer code was influenced by the following github repo and tensorflow tutorial. 
- https://github.com/hwalsuklee/tensorflow-style-transfer
- https://www.tensorflow.org/tutorials/generative/style_transfer

## Introduction
Melanoma, a form of skin cancer that arises from melanocytes, forms 75% for skin cancer related deaths in the US (Davis et al., 2019). Nevertheless, early diagnosis and classification of melanomas require clinical, dermoscopic, and molecular data and has challenges due to skin pigmentation differences in the patients, making it a growing field in Dermatology (Darmawan et al., 2019). According to Wolf et al (2013), machine learning models that are designed to tackle the melanoma detection problem remain highly inaccurate and yield approximately 30% incorrect predictions (Darmawan et al., 2019; Wolf et al., 2013). This has been causing inaccurate results, false trust in patients to these detection softwares, and even delayed melanoma detection (Wolf et al., 2013).
 
The use of deep learning models such as convolutional neural networks are found to be more accurate in identifying such cancers. However since the melanoma prevalence in Caucasian backgrounds is at least 50% greater, the majority of these deep learning models train on data that is skewed toward lighter skinned images (Raimondi, Suppa & Gandini, 2020). This underrepresentation in the machine learning models are causing high biased outcomes, unable to classify melanoma patients that have darker skin tones. (Guo et al., 2021).
 
Thus, our aim is to utilize deep learning for correcting model bias. By using a style transfer model architecture using convolutional neural networks (CNNs), we are able to take darker skin images and images of skin lesions to produce images of darker skin lesions that could be used for future deep learning model training. 
 
 
## Methodology
Dataset
In order to complete the style transfer model architecture, two different datasets are required: light skin melanoma images describing the feature of the skin lesions, and diverse skin background images describing the style of the background to incorporate into the final image result. The former dataset was compiled by the International Skin Imaging Collaboration (ISIC) to produce 2357 images of malignant and benign skin abnormalities with classification of 9 different cancer subtypes: actinic keratosis, basal cell carcinoma, dermatofibroma, melanoma, nevus, pigmented benign keratosis, seborrheic keratosis, squamous cell carcinoma and vascular lesion. The second dataset was web-scraped to identify darker-skinned image backgrounds to use for style-transfer. 
 
## Preprocessing
Preprocessing was done to normalize image data size and file type. Additionally, the data was shuffled for more accurate training and testing purposes.
 
## Model - CNN
The model, built using TensorFlow, consists of two training schemes and is based on the architecture of Gatys et. al. (2016)[1]. A convolutional neural network (CNN) was first trained to classify different skin lesions to different types of skin cancers. This was done so that the convolutional layers would learn to focus on important features of these images. The structure of this CNN consisted of 5 convolution “blocks” which were made up of two or three 2D convolution layers (with ReLU activation), dropout and batch normalization after each convolution, and max pooling at the end of each block. The CNN had a similar architecture to the VGG models with 5 CNN blocks that consisted of 2 or 3 convolutional layers followed by a max pooling layer. For each subsequent CNN block, we added more filters and arrived at the architecture: 8 → 32  → 64 → 128 → 256 (Fig. 1). By increasing the number of filters in the CNN architecture, the program first learned to pick up larger regions while towards the end of the model, it focused on more finite details. Furthermore, this model contains only 5 million parameters, which is relatively small for a CNN architecture and allows for faster training time. After the fifth convolution block, the logits were passed through three dense layers which result in a probability distribution over the 9 subcategories of skin cancers. Categorical cross-entropy loss was used to calculate the loss and the gradients were applied using the Adam optimizer. After training the layer weights are saved. 
