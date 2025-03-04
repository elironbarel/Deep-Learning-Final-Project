# Deep Learning Project: Identifying a Specific Dog

## Project Overview
This deep learning project focuses on identifying a specific dog based on images taken from various angles and at different ages. The model is trained using a convolutional neural network (CNN) to distinguish images of the target dog from other dogs in the dataset.

## Dataset
The project uses the **Stanford Dogs Dataset**, which contains **20,580 images** of dogs from different breeds. The dataset has been repurposed to train a binary classifier that differentiates between images of the specific dog and all other dogs.

### Data Preprocessing
- Images are **resized** to a fixed size (e.g., 150x150).
- **Normalization** is applied to pixel values to enhance model performance.
- **Data augmentation** is performed for the specific dog’s images, including:
  - Rotation
  - Horizontal flipping
  - Zoom and brightness adjustments

## Model Architecture
The model is implemented using **PyTorch** and follows a CNN-based approach:
- **Convolutional Layers**: Extract spatial features.
- **Max-Pooling Layers**: Reduce dimensions while retaining key information.
- **Fully Connected Layers**: Map extracted features to classification output.
- **Dropout Layers**: Prevent overfitting.
- **Sigmoid Activation**: Outputs the probability of the image being the target dog.

## Training
- **Loss Function**: Binary Cross-Entropy Loss (BCELoss)
- **Optimizer**: Adam optimizer with a learning rate of **0.001**
- **Epochs**: 10
- **Batch Size**: 32
- **GPU Acceleration**: Utilized for faster training

## Evaluation
- The trained model is evaluated using:
  - **Accuracy**
  - **Precision, Recall, and F1-score**
  - **Confusion Matrix**
- The validation dataset is used to monitor performance and avoid overfitting.

## Results
- The final model achieves **high accuracy** in distinguishing the specific dog from other breeds.
- Training and validation loss curves are analyzed to ensure proper learning without overfitting.
- Feature maps and learned filters are visualized to understand what the CNN is focusing on.

## Installation and Usage

### 1. Clone the Repository
```sh
git clone https://github.com/your-repo/deep-learning-dog-identification.git
cd deep-learning-dog-identification

### 2. **Install Dependencies**
```sh
pip install -r requirements.txt

### 3. **Download and Prepare the Dataset**

Ensure you have the **Stanford Dogs Dataset** and the **specific dog images** stored in the appropriate directories.

### 4. **Train the Model**
```sh
python src/train.py

### 5. **Evaluate the Model**
```sh
python src/evaluate.py

### 6. **Test on New Images**

Run the script to classify a new image:

```sh
python src/predict.py --image test_image.jpg


## Project Structure
```bash
/deep-learning-dog-identification
│── data/
│   ├── specific_dog/       # Target dog images
│   ├── all_dogs/           # Other dogs dataset
│   ├── augmented/          # Augmented images of the target dog
│── models/
│   ├── model.pth           # Saved trained model
│── src/
│   ├── dataset.py          # Dataset loading and preprocessing
│   ├── model.py            # CNN architecture
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   ├── predict.py          # Image classification script
│── notebooks/
│   ├── analysis.ipynb      # Data exploration and model performance analysis
│── README.md               # Project documentation
│── requirements.txt        # Python dependencies

## Future Improvements

- Implement transfer learning using a pre-trained CNN (e.g., ResNet or VGG16).
- Experiment with different architectures to enhance accuracy.
- Deploy the model as a web application for real-time image classification.

## Contributors

- Eliron Barel
- Ohad Maymon

For questions or suggestions, feel free to open an issue in the repository.
