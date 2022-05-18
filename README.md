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
