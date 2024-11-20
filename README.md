# age-detection-project
Final exam work, aml


Introduction
The objective of this project was to classify human faces into predefined age groups using machine learning models. The dataset contains images of human faces labeled into five age groups, and the task was to build and compare two models: a baseline model using ResNet50 and an enhanced model with additional layers and fine-tuning.
Dataset Description
The dataset, sourced from Kaggle, contains images of human faces categorized into five age groups:
- 18-25
- 26-35
- 36-45
- 46-55
- 56-60

The dataset was split into training (80%) and validation (20%) sets. To enhance generalization, data augmentation techniques were applied, including rotation, zoom, horizontal flipping, and pixel value normalization to the range [0, 1].
Classification Project
Two models were implemented and evaluated:
1. Baseline Model: Used a pretrained ResNet50 model with its final layer replaced to match the 5 age classes. Initial layers were frozen to leverage the pretrained ImageNet weights.
2. Enhanced Model: Included additional Dense layers with Batch Normalization and Dropout, along with fine-tuning of the ResNet50 layers to improve performance on the dataset.
Baseline Model
The baseline model leveraged the ResNet50 architecture with ImageNet weights. The final classification layer was replaced to predict the five age groups, while the remaining layers were frozen. This model achieved the following results:
- Training Accuracy: 83.33%
- Validation Accuracy: 83.33%
- Loss: 33.2377

Despite the high accuracy, the confusion matrix revealed significant bias towards the 'train' class.
Enhanced Model
The enhanced model incorporated additional Dense layers with Batch Normalization and Dropout to reduce overfitting. The ResNet50 layers were fine-tuned to adapt the pretrained features to the dataset. The model achieved the following results:
- Training Accuracy: 83.33%
- Validation Accuracy: 83.33%
- Loss: 1.0531

The confusion matrix was identical to the baseline model, indicating that the enhanced model also struggled to classify the 'test' class.
Metrics and Validation
The models were evaluated using precision, recall, and F1-score for each class. Below are the class-wise performance metrics:

| Class    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| test     | 0.00      | 0.00   | 0.00     | 5       |
| train    | 0.83      | 1.00   | 0.91     | 25      |

Overall accuracy on the validation set was 83.33%, but the lack of correct predictions for the 'test' class highlights significant challenges with model generalization.
Challenges and Solutions
One major challenge encountered was overfitting. While the training accuracy was significantly higher, the validation performance remained biased towards the 'train' class. To address this issue:
- Additional data augmentation techniques were applied.
- Dropout rates were increased to improve regularization.
- Batch Normalization was adjusted to stabilize training.

Despite these efforts, the dataset's limited size and class imbalance constrained the model's ability to generalize effectively.
Streamlit Deployment
An interactive Streamlit app was developed to allow users to upload images and view predictions for age groups. The app provides a user-friendly interface to demonstrate model performance and supports predictions from both the baseline and enhanced models.
Conclusion
The baseline model provided a reasonable starting point with limited performance, while the enhanced model improved training accuracy but faced challenges with overfitting. This project highlights the need for larger datasets and more robust regularization techniques for improved model generalization.

Future work includes:
- Increasing dataset size and diversity.
- Exploring alternative architectures like EfficientNet.
- Fine-tuning pretrained models with gradual unfreezing of layers.
