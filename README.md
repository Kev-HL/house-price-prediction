# House Price Prediction

This project tackles a real-world regression problem: predicting house sale prices based on property features.  
The goal is to explore and compare different modeling approaches using structured tabular data, with a strong focus on EDA, feature engineering, and model evaluation.

The models will predict housing sale prices using data from the **Kaggle** competition [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).  


## Project Overview

The main notebook **housing_full_analysis.ipynb** includes the full end-to-end workflow: EDA, cleaning, feature engineering and selection, model training, evaluation using RMSE, and result interpretation.  
The models trained include:  
- Ridge (linear)  
- Random Forest  
- Catboost regressor (gradient boosting)  
- Multilayer Perceptron (MLP)  
- Multi-Branch MLP (parallel dense paths with concatenation)

A second, streamlined notebook **housing_submission.ipynb** is provided to reproduce final results with only the selected features and best-performing model.


## Key Results

- `Best model:` Catboost regressor  
- `RMSE (log scale):` 0.1278
- `Top features:` TotalSF, QualityIndex, TotalFinSF, KitchenQual  


## Dataset

The Ames Housing dataset will be used, provided as part of the Kaggle competition:  
*Anna Montoya and DataCanary. House Prices - Advanced Regression Techniques. Kaggle, 2016.*  
The dataset is used strictly for non-commercial, educational purposes.

The files **train.csv** and **test.csv** which contain the data, as well as the text file **data_description.txt** can be found on the competitions page.  


## Tech Stack

- Python, Jupyter Notebooks
- Numpy, Pandas, matplotlib, seaborn, statsmodels
- Scikit-Learn, Tensorflow, Tensorflow Decision Forests (TFDF), Catboost
- Git for version control


## Folder Structure

- `data/`: raw and processed datasets
- `notebooks/`: Jupyter notebooks for EDA and modeling
- `src/`: Python scripts for reusable code
- `models/`: saved trained models
- `outputs/`: prediction results and other logs


## How to Run

Run all code sequentially from the Jupyter notebooks inside **/notebooks**.  
The project uses relative paths and assumes the dataset files are placed inside the **/data/raw** folder.  
See **requirements.txt** for dependencies.