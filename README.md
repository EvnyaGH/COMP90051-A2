# Sentiment Classification Research Project

## Research Question

**How do different text representation methods (TF-IDF, Word2Vec, ELECTRA) perform across increasing model complexity levels (LogReg, BiLSTM, ELECTRA) for sentiment classification?**

## Project Overview

This project implements a complete machine learning research pipeline for sentiment classification using the IMDB movie review dataset. The research investigates how different text representation methods perform across increasing model complexity levels, with a focus on preventing data leakage through proper experimental design.

### Key Research Components

1. **Feature Representations**:

   - **TF-IDF**: Word and character n-grams with 50k word features
   - **Word2Vec**: Skip-gram model with average pooling (100-dim vectors)
   - **ELECTRA**: Contextual embeddings from pre-trained transformer

2. **Model Complexity Levels**:

   - **Simple**: Logistic Regression (linear classifier)
   - **Medium**: BiLSTM (bidirectional recurrent neural network)
   - **Complex**: ELECTRA (transformer-based fine-tuning)

3. **Experimental Design**:
   - 3-fold outer cross-validation for final evaluation
   - 2-fold inner cross-validation for hyperparameter tuning
   - Learning curves: F1 vs training set size (10%, 20%, ..., 100%)
   - All metrics computed from scratch

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd COMP90051-A2

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 1.1. Data Preparation

The experiment automatically processes the raw IMDB Dataset.csv file. Place your raw data file at:

```
data/IMDB Dataset.csv
```

The system will automatically:

- Clean and normalize text data
- Map sentiment labels to binary format
- Create balanced 10k subset for fast mode
- Generate processed files: `imdb_clean.csv` and `imdb_balanced_10k.csv`

### 2. Run Complete Experiment

#### Basic Usage

```bash
# Run full experiment (default mode)
python run_experiment.py

# Run in fast mode (reduced dataset, fewer CV folds, minimal hyperparameters)
python run_experiment.py --fast

# Include learning curves in the experiment
python run_experiment.py --include-learning-curves

# Run fast mode with learning curves
python run_experiment.py --fast --include-learning-curves
```

#### Advanced Usage

```bash
# Specify custom raw data path
python run_experiment.py --raw-data-path "path/to/IMDB Dataset.csv"

# Combine multiple options
python run_experiment.py --fast --include-learning-curves --data-dir data --results-dir results_fast --random-state 42
```

#### Command Line Arguments

- `--fast`: Run in fast mode (reduced dataset, fewer CV folds, minimal hyperparameters)
- `--include-learning-curves`: Include learning curves in the experiment
- `--data-dir`: Directory containing dataset files (default: "data")
- `--results-dir`: Directory to save results (default: "results")
- `--random-state`: Random seed for reproducibility (default: 42)
- `--raw-data-path`: Path to raw IMDB Dataset.csv file (default: "data/IMDB Dataset.csv")

This will execute the full experimental pipeline:

1. **Data Preparation**: Automatically processes raw IMDB Dataset.csv using `prepare_dataset.py`
2. **Hyperparameter Tuning**: Cross-validation with hyperparameter optimization for all models
3. **Learning Curves**: F1 score vs training set size (if enabled)
4. **Visualizations**: Model comparison charts and learning curves
5. **Results**: Comprehensive performance metrics and summaries

### 3. Test Set Evaluation

To detect overfitting and get realistic performance estimates:

```bash
# Run test set evaluation (uses most recent experiment results)
python run_test_evaluation.py

# Specify custom data path
python run_test_evaluation.py --data-path "data/imdb_clean.csv"

# Use specific results file
python run_test_evaluation.py --results-file "experiment_results_20250115_123456.json"

# Custom test split and random seed
python run_test_evaluation.py --test-size 0.3 --random-state 123

# Custom results directory
python run_test_evaluation.py --results-dir "custom_results"
```

#### Command Line Arguments

- `--data-path`: Path to clean dataset file (default: "data/imdb_clean.csv")
- `--results-dir`: Directory containing experiment results (default: "results")
- `--results-file`: Specific results file to use (default: most recent experiment*results*\*.json)
- `--test-size`: Fraction of data to use for testing (default: 0.2)
- `--random-state`: Random seed for reproducibility (default: 42)

This will:

1. Split data into train/test sets (80/20 by default)
2. Train models on train set with best CV parameters
3. Evaluate on test set
4. Compare train vs test performance to detect overfitting

### 4. Expected Outputs

- **Console**: Progress updates and final performance summary
- **results/experiment*results*\*.json**: Raw experimental data
- **results/experiment_summary.txt**: Performance summary table
- **results/model_comparison.png**: Bar chart comparing F1 scores
- **results/learning_curves.png**: Learning curves for all models (if enabled)
- **results/model_performance_bar.png**: Detailed model performance visualization
- **results/model_summary.csv/md**: Comprehensive model comparison tables

### Evaluation Metrics

All metrics implemented from scratch:

- **Accuracy**: Overall correctness
- **Precision**: Macro, micro, weighted averages
- **Recall**: Macro, micro, weighted averages
- **F1-Score**: Macro, micro, weighted averages
- **Confusion Matrix**: Per-class performance
- **Classification Report**: Comprehensive summary

### Learning Curves

- **Training sizes**: 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%
- **Metric**: F1-Score (macro average)
- **Error bars**: Standard deviation across CV folds
- **Purpose**: Analyze data efficiency and model learning patterns

## Expected Research Insights

The experiment addresses several key research questions:

1. **Feature Representation Effectiveness**:

   - How do TF-IDF, Word2Vec, and ELECTRA compare?
   - Which representation works best for sentiment classification?

2. **Model Complexity Impact**:

   - Does increasing model complexity improve performance?
   - What are the trade-offs between complexity and performance?

3. **Data Efficiency**:

   - How much training data do different models need?
   - Which models benefit most from additional data?

4. **Feature-Model Compatibility**:
   - Which feature-model combinations work best?
   - Are there optimal pairings for sentiment classification?

## Project Structure

```
COMP90051-A2/
├── data/                          # Data directory
│   ├── IMDB Dataset.csv          # Raw IMDB dataset (place here)
│   ├── imdb_clean.csv            # Processed clean data (auto-generated)
│   ├── imdb_balanced_10k.csv     # Balanced 10k subset (auto-generated)
│   └── features/                 # Extracted features (auto-generated)
├── src/                          # Source code
│   ├── prepare_dataset.py        # Data preparation module
│   ├── extract_features.py       # Feature extraction
│   ├── models/                   # Model implementations
│   │   ├── logistic_regression.py
│   │   ├── bilstm_sentiment.py
│   │   └── electra_sentiment.py
│   ├── experiments/              # Experimental pipeline
│   │   ├── experimental_pipeline.py
│   │   ├── hyperparameter_tuning.py
│   │   ├── learning_curves.py
│   │   └── test_evaluation.py
│   └── core/                     # Core utilities
│       ├── metrics.py
│       └── cross_validation.py
├── results/                      # Results directory (auto-generated)
├── run_experiment.py             # Main experiment script
├── run_test_evaluation.py        # Test evaluation script
└── requirements.txt              # Dependencies
```

## License

This project is part of COMP90051 Statistical Machine Learning coursework at the University of Melbourne.
