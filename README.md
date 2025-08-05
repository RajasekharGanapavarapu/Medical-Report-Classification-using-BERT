# Medical Report Classification Using BERT

This project demonstrates how to classify medical reports into different categories using a BERT-based transformer model. It leverages Hugging Face’s `transformers` library, PyTorch, and a synthetic dataset of medical reports. The notebook provides a step-by-step workflow from data ingestion and preprocessing to model training, evaluation, and inference.

## Features

- **Text Classification:** Categorizes medical reports (e.g., Radiology, Cardiology, Discharge Summary, Pathology).
- **BERT Model:** Utilizes `DistilBERT` for sequence classification.
- **Transfer Learning:** Freezes base BERT layers and fine-tunes the classifier head.
- **Metrics:** Outputs precision, recall, F1-score, and support for each class.
- **Inference:** Predicts the category of a new medical report with confidence scores.

## Dataset

- Format: Excel (`.xlsx`) file with columns: `report` (text), `label` (category).
- Example: `medical_reports_1000.xlsx`  
Upload your dataset when prompted in the notebook.

## Installation

Run the following command at the start of the notebook:
```python
!pip install -q transformers datasets scikit-learn pandas openpyxl tqdm
```

## Usage

1. **Upload Data:**  
   The notebook will prompt you to upload the medical reports Excel file.

2. **Data Preprocessing:**  
   - Labels are mapped to integers.
   - Tokenization using DistilBERT tokenizer.
   - Train/test split (80/20).

3. **Model Training:**  
   - One epoch training for demonstration.
   - Uses AdamW optimizer and linear learning rate scheduler.

4. **Evaluation:**  
   - Generates a classification report with metrics for each class.

5. **Inference:**  
   - Provides a function to predict the category of new report text.

## Sample Outputs

- Classification Report with metrics (precision, recall, F1-score).
- Inference example:
  ```
  🧪 Inference:
  Report: ECG shows ST elevation with signs of infarction.
  Predicted: Radiology (Confidence: 0.27)
  ```

## Customization

- Adjust training epochs and batch size for improved performance.
- Replace DistilBERT with another transformer model if desired.
- Expand the dataset for real-world applications.

## Requirements

- Python 3.x
- Jupyter/Colab environment
- Packages: `transformers`, `datasets`, `scikit-learn`, `pandas`, `openpyxl`, `tqdm`, `torch`

## License

This project is licensed under the Apache License 2.0.

## Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Google Colab](https://colab.research.google.com)
- [PyTorch](https://pytorch.org)
