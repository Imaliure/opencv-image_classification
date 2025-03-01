
# Footballer Classification

This project is aimed at classifying images of football players using computer vision techniques. It uses a **Local Binary Patterns Histograms (LBPH)** face recognizer model from OpenCV to train on images of football players, classify them, and evaluate the model's performance.

## Project Overview

The application classifies images of football players from different classes, using LBPH for image classification. The dataset is split into training and testing sets, and the model is trained on the training set and evaluated on the testing set. After training, the model is saved and used for predictions on new images.

## Features
- **Training**: Uses LBPH face recognizer from OpenCV to train a model with football player images.
- **Prediction**: Can classify a new image based on the trained model.
- **Evaluation**: Uses accuracy, classification report, and confusion matrix to evaluate model performance.

## Requirements

To run this project, you'll need the following libraries:

- OpenCV (`cv2`)
- NumPy
- scikit-learn
- Matplotlib
- Seaborn

Install the dependencies with:

```bash
pip install opencv-python numpy scikit-learn matplotlib seaborn
```

## Dataset

The dataset used for training and testing is a collection of images of football players. Due to the large size of the dataset, it has not been included in this repository. You can download the dataset from the following link:

[Dataset Link](YOUR_DATASET_LINK_HERE)

Make sure to place the dataset in the directory structure as described in the script (`football_golden_foot` folder).

## Model Training

### How to Train the Model:
1. Place the football player images in the appropriate directory.
2. The dataset should be organized into folders, with each folder representing a football player and containing their images.
3. Run the script `footballer_classification.py` to train the model.

### Example Usage:
To classify a new image, call the `classify_image()` function:

```python
classify_image('path/to/your/image.jpg')
```

The model will predict the label (football player) and provide the confidence score.

## Model Performance

After training, the script will print the accuracy score, classification report, and confusion matrix to evaluate the model's performance.

### Example Output:
```
Model accuracy: 92.50%

Classification Report:
              precision    recall  f1-score   support

    Messi         0.95      0.94      0.94         25
   Ronaldo        0.90      0.91      0.90         25
    ...
   
Confusion Matrix:
(Plot will be shown here)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
