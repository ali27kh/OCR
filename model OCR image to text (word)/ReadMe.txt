
"This model requires more data to deliver optimal results. 
You can build your own model by collecting data specific to your language."
***************************************************************************
Data Architecture Explanation:
The dataset is organized into a hierarchical folder structure for both training and testing:

Training Data (data folder):

This folder contains subfolders, where each subfolder represents a class label (e.g., letters, numbers, or other symbols).
Each subfolder contains multiple images corresponding to that specific class.
Testing Data (testing_data folder):

Similarly, this folder is structured like the training data, with subfolders representing class labels and images corresponding to those labels.
Image Preprocessing:

Each image is read using cv2 and resized to a fixed dimension of 64x64 pixels.
The pixel values are normalized by dividing by 255 to scale them between 0 and 1.
All processed images are stored in a list, and their corresponding class labels are extracted from folder names.
Label Encoding:

Class labels (strings) are encoded into numerical values using LabelEncoder, which maps each unique label to an integer.
Data Shuffling:

The training dataset (X and y) is shuffled to ensure randomness and avoid bias during training.
Model Architecture Explanation:
The model is a Convolutional Neural Network (CNN) implemented using the Keras Sequential API. Its purpose is to classify images into one of the predefined classes.

Input Layer:

The input shape is (64, 64, 3), representing RGB images with a resolution of 64x64 pixels.
Convolutional Layers:

The network includes 4 convolutional layers:
1st Layer: 16 filters of size (3x3) with ReLU activation.
2nd Layer: 32 filters of size (3x3) with ReLU activation.
3rd Layer: 64 filters of size (3x3) with ReLU activation.
4th Layer: 128 filters of size (3x3) with ReLU activation.
Pooling Layers:

Each convolutional layer is followed by a MaxPooling2D layer, which reduces spatial dimensions (height and width) by half, helping to downsample the feature maps and reduce computational complexity.
Flatten Layer:

After the convolutional and pooling layers, the output is flattened into a 1D vector to be fed into the fully connected layers.
Dense Layers:

First Dense Layer: 128 units with ReLU activation.
Second Dense Layer: 64 units with ReLU activation.
Output Layer: 36 units (one for each class), with a softmax activation to output probabilities for each class.
Loss Function:

The model uses sparse_categorical_crossentropy as the loss function, which is suitable for multi-class classification tasks when labels are integer-encoded.
Optimizer:

The optimizer is Adam, which adapts the learning rate during training for faster convergence.
Metrics:

The model tracks accuracy during training and validation.
Model Validation and Testing:
The model is trained for 10 epochs with a batch size of 25.
A validation split of 20% is used to monitor the model's performance on unseen data during training.
Post-training, the model is evaluated on the testing dataset, where the loss and accuracy are reported.
Confusion Matrix:
Predictions are made on the testing dataset, and the predicted classes are compared with the true labels.
A confusion matrix is plotted using seaborn to visualize the model's performance across all classes.
Output and Saving:
The model is saved in the keras format as OCR_model.keras.
The class labels (class_names) are saved as a .npy file for future use in decoding predictions.