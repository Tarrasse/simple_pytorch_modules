# simple_pytorch_modules
implementation of some custom machine learning and deep learning modules and algorithms 

## Notebooks
- Image classifier - CIFAR.ipynb 
- Logistic Regression.ipynb


### 1 - Image classifier - CIFAR.ipynb 
in this notebook i'll built a convolutional neural network and trained it on CIFAR-10 dataset 

### 2 - Logistic Regression.ipynb
in this notebook i'll build a custom Classifire with softmax instead of sigmoid activation, using only Torch tensors. 

**steps**
- [x] implementing the algorithm with auto-grade and pre-implemeted gradient descent.
- [x] implementing custom gradient descent. 
- [x] implementing the algorithm with custom gradient descent.

after comparing the results of the custom GD against the pre-implemented GD, the pre-impelemented GD took nearly twice as much time as the custom GD because it does alot unneeded computation to genralize to a wide range of problems but my algorithm done only useful comutations.
