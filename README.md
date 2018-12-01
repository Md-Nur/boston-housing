# Udacity: Machine Learning Foundation Nanodegree

## Model Evaluation and Validation Project: Predicting Boston Housing Prices

### Install

This project requires **Python** and the following Python libraries installed:

- [NumPy](https://www.numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [Jupyter Notebook](https://ipython.org/notebook.html)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](https://www.anaconda.com/download/) distribution of Python, which already has most of the above packages and more included or if you don't want to install a huge number of packages then you can try [Miniconda](https://conda.io/miniconda.html) and then install the above packages.

### Code

Code is in the [boston_housing.ipynb](boston_housing.ipynb) notebook file. Also required is the included [visuals_md.py](visuals_md.py) python file which contains some modified code for model visualizations and the [housing.csv](housing.csv) dataset file.

### Run

In a terminal or command window, navigate to the top-level project directory `boston_housing/` (that contains this README) and run one of the following commands:

```bash
jupyter notebook boston_housing.ipynb
```

or

```bash
ipython notebook boston_housing.ipynb
```

This will open the Jupyter Notebook software and project file in your browser.

### Project Report

Report can be viewed at [**https://maneeshd.github.io/boston-housing/**](https://maneeshd.github.io/boston-housing/).

### Data

The modified Boston housing dataset consists of 489 data points, with each datapoint having 3 features. This dataset is a modified version of the Boston Housing dataset found on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Housing).

**Features**
1.  `RM`: average number of rooms per dwelling
2. `LSTAT`: percentage of population considered lower status
3. `PTRATIO`: pupil-teacher ratio by town

**Target Variable**
4. `MEDV`: median value of owner-occupied homes

