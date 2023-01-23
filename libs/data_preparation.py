
# DATA PRE-PROCESSING
# # 1. Outliers
# # 2. Missing Values
# # 3. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
# # 4. Feature Scaling


# Import Libraries
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.neighbors import LocalOutlierFactor

# EDA module:
from libs.exploratory_data_analysis import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)

##########################################################
###################### 1. Outliers #######################
##########################################################


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    """
    Appoints the lower and upper thresholds by use of quantile
    Parameters
    ----------
    dataframe
    col_name
    q1: first quantile percentage
    q3: third quantile percentage

    Returns
    -------
    low_limit and up_limit thresholds as tupple
    """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name, q1=0.05, q3=0.95):
    """
    Checks if there is an outlier and return bool value
    Parameters
    ----------
    dataframe
    col_name

    Returns
    -------
    outlier existance as bool
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def grab_outliers(dataframe, col_name, index=False, q1=0.05, q3=0.95, print_out=False) -> object:
    """
    Determines the index values of the outliers of the specified column in the dataframe
    Parameters
    ----------
    dataframe
    col_name
    index

    Returns
    -------
    Index values of the outliers
    """
    low, up = outlier_thresholds(dataframe, col_name, q1, q3)
    if print_out:
        if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
            print(dataframe[((dataframe[col_name] < low)
                | (dataframe[col_name] > up))].head())
        else:
            print(dataframe[((dataframe[col_name] < low)
                | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[(
            (dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


def remove_outlier(dataframe, col_name, q1=0.05, q3=0.95):
    """
    Removes the the outliers of the specified column in dataframe
    Parameters
    ----------
    dataframe
    col_name

    Returns
    -------
    object
    dataframe without outliers
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    df_without_outliers = dataframe[~(
        (dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers


def replace_with_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    """
    Replaces the outliers of the specified column with the lower or upper thresholds
    Parameters
    ----------
    dataframe
    col_name

    Returns
    -------
    No return
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
    dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit


def lof_remove(dataframe, n_neighbors=20):
    # her bir gözlemin bütün değişkenlerinin oluşturduğu çok bpyutlu uzaydaki noktanın, aynı uzaydaki
    # en yakın 20 komşu noktası ile uzaklığını değerlendiren bir model örneği oluşturulur 
    clf = LocalOutlierFactor(n_neighbors=20) 
    # model parametrik olarak verilen dataframe'e uygulanır. Her bir gözlemin 20 en yakın komşu noktasına
    # uzaklığı, bu diğer komşu noktaların birbirilerine göre uzaklığıyla karşılaştırılır. Outlier olan
    # gözlemler -1 (negatif) ile işaretlenir.
    arr = clf.fit_predict(dataframe)
    # negatif outlier factor değeleri elde edilir
    dataframe_scores = clf.negative_outlier_factor_
    dataframe_scores[0:5]
    np.sort(dataframe_scores)[0:5]

    scores = pd.DataFrame(np.sort(dataframe_scores))
    scores.plot(stacked=True, xlim=[0, 20], style='.-')
    plt.show()

    th = knee_locater(scores.index, scores[0])
    th = np.sort(dataframe_scores)[th]

    return dataframe.drop(axis=0, labels=dataframe[dataframe_scores < th].index)

##########################################################
################### 2. Missing Values ####################
##########################################################


def missing_values_table(dataframe, na_name=False):
    """
    Prints out missing value counts and ratios of the columns and returns the name of the columns with missing value
    Parameters
    ----------
    dataframe
    na_name

    Returns
    -------
    The name of the columns with missing value as a list
    """
    na_columns = [
        col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() /
             dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)],
                           axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


def missing_vs_target(dataframe, target, na_columns):
    """
    Prints out the target variable mean and count of the missing values
    Parameters
    ----------
    dataframe
    target
    na_columns

    Returns
    -------
    No return
    """
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


def missing_value_fill_knn(dataframe, cols, miss_cols):
    dff = pd.get_dummies(dataframe[cols], drop_first=True)
    dff, scaler = minmax_scaling(dff, dff.columns)

    imputer = KNNImputer(n_neighbors=5)
    dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)

    dff = minmax_reverse_scaling(dff, dff.columns, scaler)

    return dff[[miss_cols]]

##########################################################
##################### 3. Encoding ########################
##########################################################


def label_encoder(dataframe, binary_col):
    """
    Encodes the column of the dataframe with numeric labels
    Parameters
    ----------
    dataframe
    binary_col

    Returns
    -------
    The label encoded column added dataframe
    """
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


def one_hot_encoder(dataframe, cat_cols, drop_first=False, dummy_na=False):
    """
    Encodes the column of the dataframe with binary labels
    Parameters
    ----------
    dataframe
    categorical_cols
    drop_first

    Returns
    -------
    The one-hot encoded columns added dataframe
    """
    dataframe = pd.get_dummies(
        dataframe, columns=cat_cols, drop_first=drop_first, dummy_na=dummy_na)
    return dataframe


def rare_analyser(dataframe, target, cat_cols):
    """
    Prints out the class count, ratio and mean respect to the target variable of the categorical columns
    Parameters
    ----------
    dataframe
    target
    cat_cols

    Returns
    -------
    No return
    """
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


def rare_encoder(dataframe, rare_perc, cat_cols):
    """
    It changes the value to "rare" if the count of "ratio of the classes in dataframe categorical columns which is
    below rare_perc" is above 1
    Parameters
    ----------
    dataframe
    rare_perc
    cat_cols

    Returns
    -------
    dataframe with rare values labeled
    """
    # 1'den fazla rare varsa düzeltme yap. durumu göz önünde bulunduruldu.
    # rare sınıf sorgusu 0.01'e göre yapıldıktan sonra gelen true'ların sum'ı alınıyor.
    # eğer 1'den büyük ise rare cols listesine alınıyor.
    rare_columns = [col for col in cat_cols if (
        dataframe[col].value_counts() / len(dataframe) < rare_perc).sum() > 1]

    for col in rare_columns:
        tmp = dataframe[col].value_counts() / len(dataframe)
        rare_labels = tmp[tmp < rare_perc].index
        dataframe[col] = np.where(dataframe[col].isin(
            rare_labels), 'Rare', dataframe[col])

    return dataframe


def useless_cols(dataframe, rare_perc=0.01):
    useless_columns = [col for col in dataframe.columns if dataframe[col].nunique() == 2 and
                       (dataframe[col].value_counts() / len(dataframe) < rare_perc).any(axis=None)]
    dataframe.drop(useless_columns, axis=1, inplace=True)

##########################################################
################# 4. Feature Scaling #####################
##########################################################


def standart_scaling(dataframe, col_name):
    """
    Standardize features by removing the mean and scaling to unit variance.
    z = (x - u) / s
    Parameters
    ----------
    dataframe
    col_name

    Returns
    -------
    Standard scaled dataframe
    """
    ss = StandardScaler()
    dataframe[col_name] = ss.fit_transform(dataframe[[col_name]])
    return dataframe


def minmax_scaling(dataframe, col_name):
    """
    Transform features by scaling each feature to a (0,1) range.

    Parameters
    ----------
    dataframe
    col_name

    Returns
    -------
    MinMax scaled dataframe
    """
    scaler = MinMaxScaler()
    dataframe[col_name] = scaler.fit_transform(dataframe[[col_name]])
    return dataframe, scaler


def minmax_reverse_scaling(dataframe, col_name, scaler):
    """
    Transform features by scaling each feature to a (0,1) range.
    Parameters
    ----------
    dataframe
    col_name

    Returns
    -------
    MinMax scaled dataframe
    """

    dataframe[col_name] = pd.DataFrame(
        scaler.inverse_transform(dataframe[col_name]))
    return dataframe


def robust_scaling(dataframe, col_name):
    """
    Scale features using statistics that are robust to outliers.
    This Scaler removes the median and scales the data according to the quantile range (defaults to IQR)
    Parameters
    ----------
    dataframe
    col_name

    Returns
    -------
    Robust scaled dataframe
    """
    rs = RobustScaler()
    dataframe[col_name] = rs.fit_transform(dataframe[[col_name]])
    return dataframe

def log_scaling(dataframe, col_name):
    """
    Scale features using log
    Parameters
    ----------
    dataframe
    col_name

    Returns
    -------
    Log scaled dataframe
    """
    rs = RobustScaler()
    dataframe[col_name] = np.log(dataframe[[col_name]])
    return dataframe

def scaling(dataframe, method):
    """
    Scale features using the specified method.
    Parameters
    ----------
    dataframe
    method: StandartScaling, MinMaxScaling, RobustScaling, LogScaling

    Returns
    -------
    Scaled dataframe
    """
    numerical_cols = grab_col_names(dataframe)[1]
    if method == "StandartScaling":
        standart_scaling(dataframe, numerical_cols)
    elif method == "MinMaxScaling":
        minmax_scaling(dataframe, numerical_cols)
    elif method == "RobustScaling":
        robust_scaling(dataframe, numerical_cols)
    else:
        log_scaling(dataframe, numerical_cols)
    return dataframe

##########################################################
################ 4. Feature Creating #####################
##########################################################


def new_feature_quantile(dataframe, col, qnum, qnames=None, header="NEW"):
    if header == "NEW":
        new_col = "NEW_" + col
    else:
        new_col = header
    if qnames == None:
        qnames = [col.lower() + '_q' + str(i) for i in range(qnum)]
    dataframe[new_col] = pd.qcut(dataframe[col], qnum, labels=qnames)


def new_feature_interval(dataframe, col, interval_value, header="NEW"):
    if header == "NEW":
        new_col = "NEW_" + col
    else:
        new_col = header
    for intrvl, val in interval_value.items():
        dataframe.loc[[var in intrvl for var in dataframe[col]], new_col] = val


def new_feature_interval_interaction(dataframe, cols, header="NEW"):
    if header == "NEW":
        new_col = "NEW" '_'.join(cols)
    else:
        new_col = header
    dataframe[new_col] = dataframe[cols].agg('_'.join, axis=1)


def new_feature_lag(dataserie, lags, header="NEW", noise=None):
    if header == "NEW":
        new_col = "NEW_" + dataserie.head().name + "_LAG_"
    else:
        new_col = header

    dataframe = pd.DataFrame()
    for lag in lags:
        dataframe[new_col+str(lag)] = dataserie.transform(lambda x: x.shift(lag) +
                                                          (0 if noise == None else random_noise(len(x), noise)))
    return dataframe


def new_feature_roll_mean(dataserie, windows, min_periods, header="NEW", noise=None):
    if header == "NEW":
        new_col = "NEW_" + dataserie.head().name + "_ROLL_MEAN_"
    else:
        new_col = header

    dataframe = pd.DataFrame()
    for window in windows:
        dataframe[new_col + str(window)] = dataserie.transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=min_periods, win_type="triang").mean() +
                                        (0 if noise == None else random_noise(len(x), noise)))
    return dataframe


def new_feature_ewm(dataserie, alphas, lags, header="NEW"):
    if header == "NEW":
        new_col = "NEW_" + dataserie.head().name + "_EWM_"
    else:
        new_col = header

    dataframe = pd.DataFrame()
    for alpha in alphas:
        for lag in lags:
            dataframe[new_col + 'ALPHA_' + str(alpha).replace(".", "") + "_LAG_" + str(lag)] = \
                dataserie.transform(lambda x: x.shift(
                    lag).ewm(alpha=alpha).mean())
    return dataframe


def random_noise(len, scale=1.6):
    return np.random.normal(scale=scale, size=(len,))
