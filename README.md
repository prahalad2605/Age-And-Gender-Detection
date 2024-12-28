# Age & Gender Classifier using Deep CNNs

## Overview
This repository implements an age and gender classification model using deep convolutional neural networks (CNNs). The project is based on the UTKFace dataset, which is a large-scale face dataset annotated with age, gender, and ethnicity. The model takes an input image of a face and predicts the age and gender of the individual.

---

## Dataset
**UTKFace Dataset**
- The UTKFace dataset consists of over 20,000 aligned and cropped face images.
- Images include annotations for:
  - **Age**: Ranging from 0 to 116 years.
  - **Gender**: Male or Female.
  - **Ethnicity**: Multiple ethnic groups.
- The dataset exhibits high variability in:
  - Pose
  - Facial expression
  - Illumination
  - Occlusion
  - Resolution

**Note**: Input images for testing must be preprocessed to ensure vertical alignment and cropping.

---

## Features
- **Deep Learning Architecture**: Uses state-of-the-art CNNs for feature extraction and classification.
- **Multitask Learning**: Simultaneously predicts age and gender.
- **Data Augmentation**: Enhances model robustness by applying transformations such as rotation, scaling, and flipping.
- **Visualization Tools**: Generates confusion matrices and loss/accuracy curves for insights.

---

## Project Structure
```
├── data/                 # Directory for storing the UTKFace dataset
├── models/               # Saved models and training checkpoints
├── notebooks/            # Jupyter notebooks for experiments
│   └── GenderClassifier.ipynb
├── src/                  # Core scripts for preprocessing, training, and evaluation
├── results/              # Output predictions, visualizations, and metrics
├── README.md             # Project documentation
└── requirements.txt      # Required Python libraries
```

---

## Getting Started

### Prerequisites
- Python 3.8 or higher
- GPU support (optional but recommended)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Pridzzy/Age-Gender-Classifier-using-Deep-CNNs.git
   cd Age-Gender-Classifier-using-Deep-CNNs
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the UTKFace dataset and place it in the `data/` directory.

---

## Usage

### Data Preprocessing
Run the preprocessing script to prepare the dataset:
```bash
python src/preprocess.py
```

### Model Training
Train the age and gender classifier:
```bash
python src/train.py --config configs/train_config.json
```

### Evaluation
Evaluate the trained model on the test set:
```bash
python src/evaluate.py --model_path models/saved_model.pth
```

### Prediction
Make predictions on new images:
```bash
python src/predict.py --image_path path/to/image.jpg
```

---

## Results
- **Age Prediction Accuracy**: `X%`
- **Gender Prediction Accuracy**: `Y%`
- Sample visualizations:
  - Confusion matrix for gender classification.
  - Distribution of age predictions.

---

## Future Enhancements
- Add ethnicity prediction as an additional task.
- Improve model robustness for edge cases (e.g., occluded faces).
- Real-time inference using optimized deployment tools.

---

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`feature/your-feature-name`).
3. Commit your changes and push the branch.
4. Open a pull request.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- **UTKFace Dataset**: A rich dataset for age and gender prediction.
- **Libraries**: TensorFlow, PyTorch, or other frameworks used in the project.

---

## Contact
For any inquiries or suggestions, please contact:
- **GitHub**: [Prahalad](https://github.com/prahalad2605)
