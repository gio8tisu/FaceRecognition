# EDCAV 2018 Face Recognition

Python code examples for EDCAV's face recognition's lab

It needs the following packages to run:
numpy matplotlib scikit-image docopt h5py imageio sklearn

Images from several individuals will be used as a training set

```
data/train/John_Doe/{image1.jpg, ..., imageN.jpg}
data/train/Jane Doe/{image1.jpg, ..., imageN.jpg}
```

### Block diagram

```
training_images --> feature_extraction --> model_construction
                                                |                                                                
                                                |                                                                
                                                v                                                                
test_images     --> feature_extraction --> classifier --> hypothesis
```


1. Preprocessing (optional): Detect and extract the faces for the images in the test dataset and correct the illumination.
2. Model construction: For each individual (different folder), create a model (see create_all_models.py). A model is a binary file packing the vector features from the training images of a given individual, along with its name and a numeric identifier.
3. Classification: After preprocessing the test image, extract its features and compare against the models features using a suitable classifier (see test_all.py)

## Operation
### Creating the models
```
python create_all_models.py data/train models/n100
```
### Performing face recognition
```
python test_all.py data/test models/n100 
```
Output:

```
Total classification error: 2.33%
[[ 56   1   0   1   0   1]
 [  0  88   0   0   0   0]
 [  0   1  18   0   0   1]
 [  2   2   0 148   0   1]
 [  1   0   0   0 145   0]
 [  0   0   0   1   0  48]]
                  precision    recall  f1-score   support

   Agnes_Marques      0.949     0.949     0.949        59
 Carles_Francino      0.957     1.000     0.978        88
JoanCarles_Peris      1.000     0.900     0.947        20
      Nuria_Sole      0.987     0.967     0.977       153
     Raquel_Sans      1.000     0.993     0.997       146
      Xavi_Coral      0.941     0.980     0.960        49

     avg / total      0.977     0.977     0.977       515

```
