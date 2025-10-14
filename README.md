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
   - 10-fold outer cross-validation for final evaluation
   - 3-fold inner cross-validation for hyperparameter tuning
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

### 2. Run Complete Experiment

```bash
python run_experiment.py
```

This will execute the full experimental pipeline:

1. Load and prepare IMDB dataset
2. Run hyperparameter tuning for all models
3. Generate learning curves
4. Create visualizations
5. Save results

### 3. Expected Outputs

- **Console**: Progress updates and final performance summary
- **results/experiment*results*\*.json**: Raw experimental data
- **results/model_comparison.png**: Bar chart comparing F1 scores
- **results/learning_curves.png**: Learning curves for all models


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


## License

This project is part of COMP90051 Statistical Machine Learning coursework at the University of Melbourne.
