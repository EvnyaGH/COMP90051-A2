COMP90051-A2: Sentiment Classification Research Project
=======================================================

QUICK START
-----------
1. Install datasets: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
2. Place raw data: data/IMDB Dataset.csv
3. Create virtual environment (recommended): 
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
4. Install dependencies: pip install -r requirements.txt
5. Run experiment: python run_experiment.py --raw-data-path "path/to/IMDB Dataset.csv" --data-dir "custom_data" # change "path/to/IMDB Dataset.csv" to your actually path to IMDB Dataset.csv
6. Test evaluation: python run_test_evaluation.py --results-file "path/to/experiment_results_20250115_123456.json" # change to your actually path to experiment results


REQUIREMENTS
------------
- Python 3.8+
- See requirements.txt for dependencies
- Place IMDB Dataset.csv in data/ directory

OUTPUTS
-------
- results/experiment_results_*.json (raw data)
- results/model_comparison.png (performance chart)
- results/learning_curves.png (learning curves)
- results/experiment_summary.txt (summary table)
- results/confusion_matrices.png (confusion matrices)
- results/model_summary.csv/md (detailed summary tables)

For detailed documentation, see https://github.com/EvnyaGH/COMP90051-A2 README.md
