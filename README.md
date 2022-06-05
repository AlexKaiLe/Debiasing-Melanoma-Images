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

![](https://github.com/AlexKaiLe/melanoma_skin_tones/blob/main/figures/fig_1.png)
Figure 1: CNN architecture

### Model - Style Transfer
The second training scheme was the style transfer (Fig. 1) which utilized the trained CNN model in different ways (Gatys et. al. 2016). To successfully transfer a style of one image (style image) to a desired image (feature image), it is necessary to be able to capture the style components and important features of an image. To do so, the CNN model is used without the dense and dropout layers. To capture style, the pooling layers were switched from max to average pooling. Next, the latent spaces (arrays) after each block are saved. To capture features, the pooling layers remain max pooling, but only the latent space of the fourth convolution block is saved. These saved latent spaces should hold deep style and feature representations of a given image. The style transfer training scheme first computes the latent space representations of the style from the “style image” and the features from the “feature image”. A third image (input image) is then passed through both style and feature CNN’s. A style loss is computed using the difference between the “input image”’s style latent spaces and the reference “style image”’s style latent spaces. Note that these latent spaces are converted to their Gram matrix representation during calculation and the loss per layer is given by Eq. 1 while the total style loss is represented by Eq. 2.

<img src="https://github.com/AlexKaiLe/melanoma_skin_tones/blob/main/figures/eq_1.png" alt="drawing" height="100"/>


Here, E<sub>l</sub> is the style loss per block, Nl is the product of the height and width of the latent space, and Ml is the number of channels of the latent space. G<sub>i,j</sub> and A<sub>i,j</sub> are the elements within Gram matrix representations of the style latent arrays from the “input image” and “style image”. 

<img src="https://github.com/AlexKaiLe/melanoma_skin_tones/blob/main/figures/eq_2.png" alt="drawing" height="100"/>

l<sub>style</sub> is the total style loss while L is the total number of saved latent spaces. The feature loss is computed from the difference between the latent space’s of the “input image” and “feature image” after the fourth convolution block. The loss is given by Eq. 3.

<img src="https://github.com/AlexKaiLe/melanoma_skin_tones/blob/main/figures/eq_3.png" alt="drawing" height="100"/>

l<sub>feature</sub> is the feature loss, F and P are the feature latent space representations of the “input image” and the “feature image”. Eq. 4 gives the total loss which is a linear combination of the style loss and feature loss. ɑ and ꞵ are the loss weights for feature and style losses, respectively. They are hyperparameters.

<img src="https://github.com/AlexKaiLe/melanoma_skin_tones/blob/main/figures/eq_4.png" alt="drawing" height="100"/>

Since the style loss is much smaller than feature loss in magnitude, due to the Gram matrices conversion, the ɑ to ꞵ ratio must be ~1E-7. The user can emphasize features or style by adjusting ɑ and ꞵ, respectively. 

Once the total loss is computed, gradients are calculated relative to the “input image” and they are applied back to the “input image” such that it is updated with new values that should be more similar to both features and style. This process is repeated for N number of iterations until the desired amount of styling is achieved. Visualizations of the loss over time and the initial and final “input image” are also produced.

![](https://github.com/AlexKaiLe/melanoma_skin_tones/blob/main/figures/fig_2.png)
Figure 2: The model of our style transfer algorithm.

## Results
The results of the style transfer algorithm were successful to a certain extent. The classification accuracy of the content CNN architecture plateaued around 44% with tuning of the hyperparameters and overall style transfer led to a constantly decreasing loss as seen in Fig 3. In training, we saw a consistent decrease in loss with a smooth increase in training accuracy to 62%. However, our testing data became more volatile and was not as smooth as the training data. The testing losses jumped around a lot and were higher than the training loss on average. The testing accuracies also jumped around a lot and were lower than the training accuracy on average Fig 3. Though this is not highly accurate in a classification sense, it is higher than accuracy produced from random guessing (1/9=11%), and the overall objective of the model was not classification, but to train a CNN enough to be able to identify important features.

![](https://github.com/AlexKaiLe/melanoma_skin_tones/blob/main/figures/fig_3.png)
Figure 3: CNN classification loss and accuracy

 Based on Fig 4, it is clear that our CNN model identified the skin lesion of the feature image, such that a general outline of the lesion was overlaid onto a new, darker-skinned style. As such, we were able to produce a model that has the ability to identify important features and merge style and feature images to generate new image data from a skewed dataset. 

![](https://github.com/AlexKaiLe/melanoma_skin_tones/blob/main/figures/fig_4.png)
Figure 4: Four different combinations of style and feature images where the style image was also the image that was iteratively modified to output a style-transfer image.

As seen by the decreasing loss in Fig. 5, the overall loss is continuously decreasing as we would expect. However, there are also large jumps in these loss values that we predicted, as well. These are because the model is trying to decrease the loss of the competing elements of style and features. As the picture takes on more style elements, the features are worsened, and vice versa. For the case of the large jumps, the gradient may have changed the image to replicate either the feature or style too much, causing the other’s loss to increase significantly.

<img src="https://github.com/AlexKaiLe/melanoma_skin_tones/blob/main/figures/fig_5.png" alt="drawing" height="500"/>

Figure 5: The loss over i iterations after the first 200 iterations. The reason the first 200 were not included was because the loss values were extremely large. Including the first 200 iterations into a single graph would have resulted in a large initial drop and a seemingly insignificant decrease afterwards.

Gatys et. al. (2016)[1] claimed that inputting an image of randomized pixel values would work. By extension, we also tried to input the “feature image” rather than the “style image”, but we got unclear results from these as shown in Fig. 6. However, in both cases, we get a constantly decreasing loss as seen in Fig. 7. We attribute the lack of change in both Fig. 4 and 5 to lack of training time and perhaps insufficient loss parameters. The computational cost to develop these images decreased our ability to produce better results. For example, these were run for 5000 iterations, taking about 20 minutes each. To produce a valid looking image, it may take many times longer. There may also be issues with the loss parameters and weights, as finding the right balance between features and style is a huge component of successful results. 

![](https://github.com/AlexKaiLe/melanoma_skin_tones/blob/main/figures/fig_6.png)
Figure 6: Trials where the initial images were either randomized values (Ex. 1) or the feature image (Ex. 2). In Ex. 1 there seems to be a slight dark patch growing in the center, but it is not significant and no style change is present. In Ex. 2, there is significant change in the pixel values, but rather than turning more brown, there are purple/green hues. 

![](https://github.com/AlexKaiLe/melanoma_skin_tones/blob/main/figures/fig_7.png)
Figure 7: The loss per iteration of Ex. 1 of Fig. 6 (left) and Ex. 2 of Fig. 6 (right)

## Challenges
While the style transfer model is successful, there is significant room for improvement. Tuning hyperparameters to find the most optimal values may improve accuracy for the CNN classification part. Second, there are decisions to improve style transfer by using the style image (background skin color) as the initial image and overlay the pattern of the lesioned image. This produced the result of identifying a general outline of the lesion on top of a diverse skin color. However, abnormalities in color and difference in resolution due to static image influence the style transfer success. In addition, the lesion outline is visible but not distinct enough to see a large enough contrast. Fixing these issues would be some of the plausible next steps to improve this model.

## Reflection
This project was successful in showing that it is possible to generate new images from existing data. Not only was the technology impressive, it also fills a unique niche in the health setting. Because health data is hard to process and attain, a method like this could be incredibly useful in being able to train new deep learning models for algorithms that may easily be implemented in real life. Especially in publicly available melanoma data, there has been a lack of dark skin samples present, leading to incorrect predictions for melanoma classification and detection models. Our model could solve this problem by generating dark skin melanoma samples and hence improving the prediction models.
 
The project itself involved many trials. This included many stages such as testing different model architectures and approaches, learning new hyperparameters, and figuring out how to transfer image styles. We tried changing the number of convolutional layers, different parameters to tune the CNN, and then using different style transfer mechanisms. In the end, we differed from the original paper we started off with to modify it for specific use with biological data, particularly melanoma data. We found utilizing the styled image as the initial starting point and overlaying the pattern of the lesion proved better in combining images. Through this process, we were able to produce a general method of transformation of images and produced data that could theoretically be used for future training data. Although the accuracy is not optimal, it is a starting point for generating further images to train melanoma prediction and classification models. 

Currently, due to the accuracy and the structure of the CNN and the style transfer models that we utilized, we have only been able to get the outline of a lesion on darkened skin, and not completely the melanoma sample on a dark skin. Thus, this project still requires a decent amount of future steps. First, rather than produce a general outline, the hope is to produce a more realistic image. Through potential color correction or tuning of model hyperparameters, the goal is to learn more features about the lesion (shape, size, color, density, contrast, etc). In addition, there could be different applications of this idea to potentially more complex methods. Rather than combine styles, we could have deep learning models learn patterns in categories such as symmetry, darkness of lesion, spread, uneven diameter, etc. These complex features that distinguish malignancies from benign lesions can then be applied to darkened skin backgrounds and generated through GANs or VAEs. This approach could also be utilized in other medical fields. These applications include using style transfer to get high-dose CT scans from low-dose scans and many other applications in both basic biology research and clinical applications (Xu, Li, & Shin, 2020). 
 
Overall, this project was a successful step towards producing new melanoma data with darker skin tones, which could be used for training future neural networks that could yield more accurate predictions for more diverse patient populations.

References
- Gatys, L. A., Ecker, A. S.  & Bethge, M. (2016) Image Style Transfer Using Convolutional Neural Networks. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2414-2423. doi: 10.1109/CVPR.2016.265.
- Davis, L. E., Shalin, S. C., & Tackett, A. J. (2019). Current state of melanoma diagnosis and treatment. Cancer biology & therapy, 20(11), 1366–1379. https://doi.org/10.1080/15384047.2019.1640032
- Darmawan, C. C., Jo, G., Montenegro, S. E., Kwak, Y., Cheol, L., Cho, K. H., & Mun, J. H. (2019). Early detection of acral melanoma: A review of clinical, dermoscopic, histopathologic, and molecular characteristics. Journal of the American Academy of Dermatology, 81(3), 805–812. https://doi.org/10.1016/j.jaad.2019.01.081
- Wolf, J. A., Moreau, J. F., Akilov, O., Patton, T., English, J. C., 3rd, Ho, J., & Ferris, L. K. (2013). Diagnostic inaccuracy of smartphone applications for melanoma detection. JAMA dermatology, 149(4), 422–426. https://doi.org/10.1001/jamadermatol.2013.2382
- Raimondi, S., Suppa, M., & Gandini, S. (2020). Melanoma Epidemiology and Sun Exposure. Acta dermato-venereologica, 100(11), adv00136. https://doi.org/10.2340/00015555-3491
- Guo, L. N., Lee, M. S., Kassamali, B., Mita, C., & Nambudiri, V. E. (2021). Bias in, bias out: Underreporting and underrepresentation of diverse skin types in machine learning research for skin cancer detection-A scoping review. Journal of the American Academy of Dermatology, S0190-9622(21)02086-7. Advance online publication. https://doi.org/10.1016/j.jaad.2021.06.884
- Xu, Y., Li, Y. & Shin, BS. (2020). Medical image processing with contextual style transfer. Hum. Cent. Comput. Inf. Sci. 10, 46 . https://doi.org/10.1186/s13673-020-00251-9
