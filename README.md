# AI-Based Fake Job Advertisement Detection & Verification System

This project is an AI-powered system designed to detect and verify fake job advertisements by analyzing visual and text features from job posters. It uses a Convolutional Neural Network (CNN) trained on a synthetic dataset of genuine and fake job posters. The text extraction is handled using Tesseract OCR.

## Features
- **Synthetic Dataset Generation**: Automatically generates a balanced dataset of genuine and fake job advertisement posters with varying visual artifacts, typography, and color schemes.
- **Image Processing & Augmentation**: Employs OpenCV for dynamic range normalization and contrast enhancement (CLAHE) to preprocess images before feeding them to the model.
- **CNN Classifier**: A 4-block Convolutional Neural Network with Batch Normalization, Max Pooling, and Dropout to detect visual layout clutter, saturation, and typographic chaos associated with scam posters.
- **Evaluation**: Outputs training curves, confusion matrix, and classification reports to evaluate the model's performance on a validation set.

## Technologies Used
- Python 3
- TensorFlow / Keras (CNN Model Building & Training)
- OpenCV (Image Processing)
- Pillow (PIL) (Synthetic Dataset Generation)
- Pytesseract (OCR for text extraction)
- Matplotlib & Seaborn (Visualization & Evaluation)
- Scikit-learn (Metrics)
- Gradio (UI Interface)

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/RohindhShan/Job_Detection.git
   cd Job_Detection
   ```

2. **Install system dependencies**:
   Tesseract OCR needs to be installed on your system.
   - For Ubuntu/Debian:
     ```bash
     sudo apt-get update
     sudo apt-get install -y tesseract-ocr libtesseract-dev
     ```
   - For Windows: Download the installer from the [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki) repository and add it to your PATH.

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

1. Open the `Job_Detection.ipynb` notebook in Jupyter Notebook or Google Colab.
2. Run the cells in order:
   - **Cell 1**: Sets up the environment and installs required libraries.
   - **Cell 2**: Generates the synthetic dataset (~1500 images) split into training and validation sets.
   - **Cell 3**: Configures the data pipelines and augmentation using OpenCV and Keras `ImageDataGenerator`.
   - **Cell 4**: Constructs, trains, and evaluates the CNN model, saving the best weights to `best_model.h5`.
3. (Optional) Run the Gradio cells at the end of the notebook (if applicable) to launch the web interface for testing the model with your own job posters.

## Project Structure
- `Job_Detection.ipynb`: The main notebook containing all the code for dataset generation, model training, and evaluation.
- `requirements.txt`: Python package dependencies.
- `dataset/`: (Generated at runtime) Contains the training and validation images for genuine and fake posters.
- `model.h5` / `best_model.h5`: (Generated at runtime) Saved model weights after training.

## License
MIT License
