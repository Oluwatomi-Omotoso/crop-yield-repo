## **Goal of this Project**

To predict expected crop yields using environmental and agricultural inputs such as rainfall, temperature, and pesticide usage.

The project covers the full data science pipeline, from raw data ingestion and exploratory data analysis through to model training and evaluation, and a deployable application.

you can view the full interactive application [here:](https/oluwatomi-omotoso-crop-yield-predictor.streamlit.app/)

---

## **Project Structure**

```
CROP YIELD PREDICTOR/
│
├── app/
│   └── app.py                        # Main application entry point
│
├── data/
│   ├── processed/
│   │   ├── algoritm_comparison_data.csv   # Benchmark results across models
│   │   └── Final_dataset.csv              # Cleaned, feature-engineered dataset
│   └── raw/
│       ├── pesticides.csv                 # Pesticide usage per crop/region
│       ├── rainfall.csv                   # Annual average rainfall data
│       ├── temp.csv                       # Temperature readings
│       └── yield.csv                      # Historical crop yield records
│
├── EDA/
│   ├── charts/
│   │   ├── India's_crops_rankings_chart.png
│   │   ├── Maize_yield_chart.png
│   │   ├── pesticide_usage_for_each_crop_a....png
│   │   ├── Potatoes_yield_chart.png
│   │   └── Wheat_yield_chart.png
│   ├── src/
│   │   ├── bivariate_analysis.py          # Feature-pair relationship analysis
│   │   ├── data_inspection.py             # Initial data profiling (Summary and Datatype Inspection)
│   │   ├── handle_missing_values.py       # Strategies to handle missing values (Filling null entries, and dropping rows or columns)
│   │   ├── missing_values.py              # Missing value detection
│   │   ├── multivariate_analysis.py       # Multi-feature correlation analysis
│   │   ├── ranking_analysis.py            # Crop/region yield rankings
│   │   └── univariate_analysis.py         # Individual feature distributions
│   └── EDA.ipynb                          # Full exploratory analysis notebook
│
├── ML/
│   ├── models/
│   │   └── NN_Models/
│   │       ├── category_mappings.pkl      # Encoded label mappings
│   │       ├── neural_net.pth             # Trained PyTorch neural network
│   │       ├── scaler.pkl                 # Input feature scaler
│   └─────── final-model.pkl               # Best performing serialised model
│   └─────── scaler-model.pkl              # Scaler paired with final model
│   ├── Notebooks/
│   │   ├── ML_building.ipynb              # Model selection and training
│   │   └── Neural_net_linear_reg.ipynb    # Neural network & regression experiments
│   └── pipelines/
│       ├── customize_dataset.py           # Dataset preparation utilities
│       ├── evaluate_algorithms.py         # Cross-model benchmarking
│       └── train_and_evaluate.py          # Training loop and evaluation metrics
│
└── README.md
```

## **Features**

- Multi-sourced data integration: Combines yield, rainfall, temperature, and pesticide records into a unified dataset

- Comprehensive EDA: Univariate, bivariate and multi-variate analyses with dedicated visualisation scripts

- Algoritm bencharking: Compares multiple ML algorithms to identify the best performer

- Neural network support: Includes a trained Pytorch model alongside classical ML approaches

- Modular pipelines: Seperates scripts for data preparation, trainning and evaluation make experimentation easy.

- Ready to run application: _app.py_ exposes predictions through a clean interface

## **Tech stack**

| Area                | Tools                             |
| ------------------- | --------------------------------- |
| Language            | Python 3                          |
| ML Framework        | scikit-learn, PyTorch             |
| Data Processing     | pandas, NumPy                     |
| Visualisation       | Matplotlib, Seaborn               |
| Notebooks           | Jupyter                           |
| Model Serialisation | pickle (`.pkl`), PyTorch (`.pth`) |

---

## Getting Started

1. Clone the repositorry

```bash
git clone https://github.com/oluwatomi-omotoso/crop-yield-predictor.git

cd crop-yield-predictor
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the application

```bash
cd ./app
streamlit run app.py
```

## Exploratory Data Analysis

The `EDA/` directory contains both a consolidated notebook and modular analysis scripts

## Model Training & Evaluation

Two modelling approaches were explored and benchmarked:

**Classiscal ML** - multiple algorithms were evaluated using `evaluate_algorithms.py`. Results are stored in `data/processed/algorthm_comparison_data.csv`. The best model is serialised as final-model.pkl

**Neural Network**: A Pytorch-based network (neural_net.pth) was trained alongside a linear regression baseline, documented in `Neural_net_linear_reg.ipynb`

**Pre-processing artefacts**: `scaler-model.pkl`, `category_mappings.pkl` are stored alongside the models to ensure consistent inference.

## Data Sources

| File             | Description                                      |
| ---------------- | ------------------------------------------------ |
| `yield.csv`      | Historical crop yield records by region and year |
| `rainfall.csv`   | Annual average rainfall data                     |
| `temp.csv`       | Average temperature readings                     |
| `pesticides.csv` | Pesticide usage by crop and country              |

Raw files are merged and cleaned into `Final_dataset.csv` during preprocessing.

```
    __ _ _
  / ____| |
 | |    | |__   ___  ___ _ __   ___
 | |    | '_ \ / _ \/ _ \ '__|/ __|
 | |____| | | |  __/  __/ |   \__ \
  \_____|_| |_|\___|\___|_|   |___/
```
