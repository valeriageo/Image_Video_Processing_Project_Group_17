# IIVP 2026 Challenge

## Notebook Queue:

1. `notebooks/data_loading_sanity_checks.ipynb`
2. `notebooks/preprocessing_and_split.ipynb`
3. `notebooks/baseline_cnn_model.ipynb`
4. `notebooks/training_and_validation.ipynb`
5. `notebooks/test_predictions_and_submission.ipynb`

### 1. Data Loading and Sanity Checks:

This notebook starts the project in the simplest possible way:

1. Load the CSV files
2. Check the class balance
3. Open a few sample images
4. Confirm the image size and grayscale format

The loading logic is located in `src/data_loading.py`

### 2. Preprocessing and Split:

This notebook prepares the data for training:

1. Load training data and split 80/20 (stratified)
2. Define data transforms: rotation, affine shifts, grayscale, resize, normalization
3. Create train and validation datasets
4. Verify transformed sample has correct shape and normalized pixel values

The preprocessing logic is located in `src/preprocessing.py`.

### 3. Baseline CNN Model:

This notebook defines the model architecture:

1. Import the baseline CNN from `src/baseline_cnn.py`
2. Instantiate the model with 10 output classes
3. Test with dummy input to verify output shape

The model is a 3-block CNN with batch normalization and dropout. This is just verification—training happens next.

### 4. Training and Validation:

This notebook trains the model:

1. Load train/validation data and create dataloaders
2. Initialize model, loss function, and optimizer
3. Train for 3 epochs, tracking loss and accuracy
4. Plot training curves and save the model checkpoint

Final accuracies are printed. The trained model is saved to `outputs/baseline_cnn.pt`.

### 5. Test Predictions and Submission:

This notebook generates the final submission:

1. Load the trained model checkpoint
2. Build the test dataset (no labels)
3. Generate predictions on test images
4. Save results to `outputs/submission.csv`

This is the final step in the pipeline.

## How To Start:

### Prerequisites:
- Python 3.9 or higher
- Jupyter Notebook or VS Code with Jupyter extension

### Setup (First Time Only):
1. Clone the repository and navigate to the project directory
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project:
1. Start Jupyter:
   ```bash
   jupyter notebook
   ```
   Or in VS Code: Open the workspace folder and open any `.ipynb` file

2. Run the notebooks in this order:
   - `notebooks/data_loading_sanity_checks.ipynb`
   - `notebooks/preprocessing_and_split.ipynb`
   - `notebooks/baseline_cnn_model.ipynb`
   - `notebooks/training_and_validation.ipynb`
   - `notebooks/test_predictions_and_submission.ipynb`

3. Run cells from top to bottom in each notebook

### Project Structure:
- `src/` - Reusable Python modules imported by notebooks
  - `baseline_cnn.py` - CNN model architecture
  - `data_loading.py` - Data loading utilities
  - `preprocessing.py` - Data transforms and dataset classes
  - `training.py` - Training and evaluation functions
  - `inference.py` - Prediction and submission utilities
- `notebooks/` - Jupyter notebooks for the pipeline
- `train/train/` - Training images (organized by class: 0-9)
- `test/` - Test images
- `*.csv` - Train/test metadata files
- `outputs/` - Generated model checkpoints and predictions

### Notes:
- The notebooks automatically detect the project root, so no manual path editing needed
- The project supports CPU, GPU (CUDA), and Apple Silicon (MPS) automatically
- Training takes ~30 seconds on a modern laptop
- Final submission saved to `outputs/submission.csv`
