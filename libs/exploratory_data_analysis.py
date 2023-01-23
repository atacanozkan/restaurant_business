
# EXPLORATORY DATA ANALYSIS
# # 1. General Exploration
# # 2. Variable Distinction

# Import Libraries
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from kneed import KneeLocator
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)

##########################################################
################ 1. General Exploration ##################
##########################################################


def load_dataset(filename, extension='.csv', sep =','):
    """
    Iports the dataset
    Parameters
    ----------
    dataset

    Returns
    -------
    dataframe
    """
    if 'csv' in extension:
        data = pd.read_csv(filename+extension, sep=sep)
    elif 'xls' in extension:
        data = pd.read_excel(filename+extension)
    elif 'pkl' in extension:
        data = pd.DataFrame(pickle.load(open(filename+extension, 'rb')))
    return data


def save_dataset(data, filename, extension='.csv'):
    """
    Iports the dataset
    Parameters
    ----------
    dataset

    Returns
    -------
    dataframe
    """
    if 'csv' in extension:
        data.to_csv(filename+extension)
    elif 'xls' in extension:
        data.to_excel(filename+extension, index=False)
    elif 'pkl' in extension:
        pickle.dump(data, open(filename+extension, 'wb'))


def dumy_dataset(nan=False):
    data = pd.DataFrame(np.array([np.arange(1, 21), np.arange(
        101, 121), np.arange(1001, 1021)]).T, columns=['A', 'B', 'C'])
    data['A_CAT'] = pd.qcut(data['A'], 5, labels=[
                            'A1', 'A2', 'A3', 'A4', 'A5'])
    data['B_CAT'] = pd.qcut(data['B'], 4, labels=['B1', 'B2', 'B3', 'B4'])
    data['C_CAT'] = pd.qcut(data['C'], 3, labels=['C1', 'C2', 'C3'])

    if nan:
        data['A'][5] = np.NaN
        data['B'][3] = np.NaN
        data['C'][8] = np.NaN
        data['A_CAT'][2] = np.NaN
        data['B_CAT'][7] = np.NaN
        data['C_CAT'][4] = np.NaN

    return data


def upper_col_name(dataframe):
    """
    Convert the letters of the columns to upper
    Parameters
    ----------
    dataframe

    Returns
    -------
    dataframe which columns converted to upper
    """
    upper_cols = [col.upper() for col in dataframe.columns]
    dataframe.columns = upper_cols
    return dataframe


def check_df(dataframe, head=5):
    """
    Prints out the shape, types, head, missing values and quantiles of the dataframe
    Parameters
    ----------
    dataframe
    head

    Returns
    -------
    No return
    """
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("################ Missing Values ################")
    print(dataframe.isnull().sum())

def knee_locater(x, y, curve='concave', direction='increasing'):
    knee = KneeLocator(x, y, curve=curve, direction=direction).knee
    return knee

    
##########################################################
############### 2. Variable Distinction ##################
##########################################################


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Determines the categorical, numerical and categorical but cardinal columns.

    Parameters
    ------
        dataframe: dataframe
                dataframe whihc inludes the columns
        cat_th: int, optional
                class threshold valuse for determining numeric but categorical variable
        car_th: int, optional
                class threshold valuse for determining categoric but cardinal variable

    Returns
    ------
        cat_cols: list
                categorical columns
        num_cols: list
                numerical columns
        cat_but_car: list
                categoric but cardinal columns

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = total columns
        cat_cols = all_cat_cols + num_but_cat - cat_but_car
        num_cols = all_num_cols - num_but_cat
    """

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")

    # cat cols
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    print(f'init cat_cols: {len(cat_cols)}')

    # num cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    print(f'init num_cols: {len(num_cols)}')

    # num but cat cols
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    print(f'num_but_cat: {len(num_but_cat)}')

    # cat but car cols
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    print(f'cat_but_car: {len(cat_but_car)}')

    # cat cols
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    print(f'final cat_cols: {len(cat_cols)}')

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    print(f'final num_cols: {len(num_cols)}')

    return cat_cols, num_cols, cat_but_car


def cat_cols_summary(dataframe, cat_cols, plot=False):
    """
    Ratio of the categorical classes in a column
    Parameters
    ----------
    dataframe
    col_name
    plot

    Returns
    -------
    No return
    """
    print("############## Frequency and Ratio #############")
    print(pd.DataFrame({"Freq": dataframe[cat_cols].value_counts(),
                        "Ratio": 100 * dataframe[cat_cols].value_counts() / len(dataframe)}).rename_axis(cat_cols))

    if plot:
        for col in cat_cols:
            sns.countplot(x=dataframe[col], data=dataframe)
            plt.show()


def num_cols_summary(dataframe, num_cols, plot=False):
    """
    Numerical variable exploration
    Parameters
    ----------
    dataframe
    numerical_col
    plot

    Returns
    -------
    No return
    """
    print("################### Describe ###################")
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40,
                 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[num_cols].describe(quantiles).T)

    if plot:
        for col in num_cols:
            dataframe[col].hist(bins=20)
            plt.xlabel(col)
            plt.title(col)
            plt.show()


def target_vs_cat_cols_summary(dataframe, target, cat_col):
    """
    Prints out the target mean of the categorical classes
    Parameters
    ----------
    dataframe
    target
    categorical_col

    Returns
    -------
    No return
    """
    print(pd.DataFrame(
        {target+"_MEAN": dataframe.groupby(cat_col).agg({target: "mean"})[target]}), end="\n\n\n")


def target_vs_num_cols_summary(dataframe, target, num_col):
    """
    Prints out the defined numeric variable mean of the target classes
    Parameters
    ----------
    dataframe
    target
    num_col

    Returns
    -------
    No return
    """
    print(pd.DataFrame(
        {num_col+"_MEAN": dataframe.groupby(target).agg({num_col: "mean"})[num_col]}), end="\n\n\n")


def high_correlated_cols(dataframe, num_cols, plot=False, corr_th=0.90):
    """
    Correlation between the variables of te dataframe
    Parameters
    ----------
    dataframe
    plot
    corr_th

    Returns
    -------
    No return
    """
    corr = dataframe[num_cols].corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(
        np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool_))
    drop_list = [col for col in upper_triangle_matrix.columns if any(
        upper_triangle_matrix[col] > corr_th)]
    if plot:
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, annot=True, cmap="RdBu")
        plt.show()
    return drop_list


def get_correlated_cols(dataframe, num_cols, plot=False, asc=False):
    """
    Correlation between the variables of te dataframe
    Parameters
    ----------
    dataframe
    plot
    corr_th

    Returns
    -------
    No return
    """
    corr = dataframe[num_cols].corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(
        np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    if plot:
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(upper_triangle_matrix, annot=True, cmap="RdBu")
        plt.show()
    return upper_triangle_matrix.unstack().sort_values(ascending=asc)
