# Action Segmentation from Human-Robot Demonstrations for Grocery Picking

This repo contains experiments and source code for work in Action Segmentation from Human-Robot Demonstrations for Grocery Picking by Sonia Mathews.
This work proposes a multimodal model that learns from human-robot demonstrations (RGB images and Haptic feedback) to apply action segmentation for pick and place systems. 

All Experiments are done in PyTorch and Pytorch Lightning.

This project contains the following files and folders

[config/](https://github.com/sonia1313/Action-Segmentation-for-Grocery-Picking/tree/master/config) contains baseline configurations for each implemented model.\
[cross-validation/](https://github.com/sonia1313/Action-Segmentation-for-Grocery-Picking/tree/master/cross-validation) contains weights a models during kfold cross validation, used to ensemble the models and take the average predictions on a separate test dataset.\
[models/](https://github.com/sonia1313/Action-Segmentation-for-Grocery-Picking/tree/master/models) contains the source code of all implemented models in this project.\
[unit_tests/](https://github.com/sonia1313/Action-Segmentation-for-Grocery-Picking/tree/master/models) tests to check preprocessing and data loader implemented in this project. \
[utils/](https://github.com/sonia1313/Action-Segmentation-for-Grocery-Picking/tree/master/utils) a collection of scripts to run accuracy metrics, dataloading, data analysis and preprocessing.

The following scripts implement kfold cross validation:
- cnn_tcn_lstm_kfold.py
- cnn_lstm_kfold.py
- cnn_tcn_kfold.py
- cnn_tcn_lstm_kfold.py
- lstm_kfold.py
- tcn_kfold.py

The following script implements a single train/val/test split to fit the model:
- lstm_test.py
- tcn_test.py

All experiments run in Google Colab:
- model_kfold.ipynb - to fit model using kfold cross validation.
- model_test.ipynb - to fit a model using a single train/val/test split on daataset.Used by author for model debugging. 


Note:
- The accuracy metrics implemented in thhis project uses source code from the study [Temporal Convolutional Networks for Action Segmentation and Detection](https://github.com/colincsl/TemporalConvolutionalNetworks/tree/master/code).
- The Temporal Convolution Network (TCN) used in this project uses source code provided from the study [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://github.com/locuslab/TCN/tree/master/TCN).




