from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import streamlit as st


def plot_for_all_cols(df: pd.DataFrame, plot_kind = 'box'):
    '''
    Plots box or hist plot for all cols of given dataframe, assuming all cols are numeric
    '''
    plt_num_cols = 2
    plt_num_rows = int(np.ceil(len(df.columns) / plt_num_cols))
    # https://discuss.streamlit.io/t/matplotlib-plots-are-blurry/1224
    fig, axes = plt.subplots(nrows = plt_num_rows, ncols = plt_num_cols, figsize = (20,60))
    plt.subplots_adjust(hspace = 0.4)

    for idx, col in enumerate(df.columns):
        plt_row_idx = int(np.floor(idx / plt_num_cols))
        plt_col_idx = idx % plt_num_cols
        axis = axes[plt_row_idx][plt_col_idx]
        if plot_kind == 'box':
            sns.boxplot(data = df[col], ax = axis)
        else:
            sns.histplot(data = df[col], bins = 100, ax = axis)
        axis.set_title(col)

    st.pyplot(fig)