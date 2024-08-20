
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
from matplotlib.scale import LogScale
from matplotlib.ticker import AutoMinorLocator
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from math import exp
import warnings
warnings.filterwarnings('ignore')
plt.rcParams["font.family"] = "Arial"
COLORS= [
        "#5E81AC",
        "#BF616A",
        "#B48EAD",
        "#EBCB8B",
        "#B48EAD",
        "#C72C41",
        "#EE4540",
        "#E3E3E3",
    ]

def adf_test(series, alpha=0.05):
    """
    Perform Augmented Dickey-Fuller (ADF) test for stationarity.

    Parameters:
    - series (pd.Series): Time series data.
    - alpha (float): Significance level for the test.

    Returns:
    - ADF test results and conclusion.
    """
    result = adfuller(series, autolag="AIC")
    p_value = result[1]

    if p_value <= alpha:
        return f"Result: Reject the null hypothesis at {alpha} significance level (p-value: {round(p_value,2)}). The time series is likely stationary."
    else:
        return f"Result: Fail to reject the null hypothesis at {alpha} significance level (p-value: {round(p_value,2)}). The time series is likely non-stationary."


def plot_acf_pacf_side_by_side(
    data,
    lags=None,
    padding=0.1,
    title="Autocorrelation and Partial Autocorrelation Functions",
):
    # Set a neutral color palette and Arial font
    sns.set_palette("gray")
    plt.rcParams["font.family"] = "Arial"

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # ACF plot
    plot_acf(data, lags=lags, ax=ax[0], color="gray")
    ax[0].set_title("Autocorrelation Function (ACF)")

    # PACF plot
    plot_pacf(data, lags=lags, ax=ax[1], color="gray")
    ax[1].set_title("Partial Autocorrelation Function (PACF)")

    # Remove spines from both x and y axes
    sns.despine(ax=ax[0], left=True, right=True, top=True, bottom=True)
    sns.despine(ax=ax[1], left=True, right=True, top=True, bottom=True)

    # Adjust layout with padding
    plt.subplots_adjust(wspace=padding)

    # Add footnote
    fig.text(0.5, -0.05, adf_test(data, 0.05), ha="center", fontsize=10, color="gray")
    plt.suptitle(title, y=1.02, fontsize=14, color="#4C566A", fontweight="bold")
    # Show the plots
    return fig


def plot_multiline_chart(
    data,
    title="Smooth Multiline Chart",
    x_label="X-axis",
    y_label="Y-axis",
    y_scale="linear",
):
    # Define the color scheme with shades of red and black
    colors = COLORS
    # Check if the number of data sets is within the supported range
    if len(data) > len(colors):
        raise ValueError(
            "Too many data sets. Maximum supported is {}".format(len(colors))
        )
    # Create a plot with a different color for each data set
    plt.figure(figsize=(10, 7))
    for i, (x_axis, y_axis, label) in enumerate(data):
        plt.plot(x_axis, y_axis, label=label, color=colors[i])
    plt.title(title, fontsize=28, fontweight="bold", loc="left", color="#4C566A", pad=20)
    # Add labels and legend
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Add mildly visible dotted grid lines in x direction
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.yscale("linear")
    if y_scale == "log":
        plt.yscale("log")
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    # Remove spines from x and y axis
    sns.despine(left=True, bottom=True)

    # Add legend
    plt.legend()

    return plt


def plot_multiscatter(
    data,
    title="Smooth Multiline Chart",
    x_label="X-axis",
    y_label="Y-axis",
    y_scale="linear",
    marker="o",
    markersize=5,
    alpha = 0.5
):
    # Define the color scheme with shades of red and black
    colors = COLORS
    # Check if the number of data sets is within the supported range
    if len(data) > len(colors):
        raise ValueError(
            "Too many data sets. Maximum supported is {}".format(len(colors))
        )
    
    def unpack_and_pad(tup):
        # Create a list from the tuple
        list_from_tuple = list(tup)
        
        # Pad the list with None until its length is 6
        while len(list_from_tuple) < 6:
            list_from_tuple.append(None)
        
        # Unpack the list into variables
        var1, var2, var3, var4, var5, var6 = list_from_tuple
        
        return var1, var2, var3, var4, var5, var6
    # Create a plot with a different color for each data set
    plt.figure(figsize=(10, 7))
    for i, d in enumerate(data):
        x_axis, y_axis, label, marker, marker_size, alpha_input = unpack_and_pad(d)
        if marker_size is not None:
            markersize = marker_size
        if alpha_input is not None:
            alpha = alpha_input
        plt.scatter(x_axis, y_axis, label=label, color=colors[i], marker=marker, s=markersize, alpha=alpha, edgecolors=None)
    plt.title(title, fontsize=28, fontweight="bold", color="#4C566A", loc="left", pad=20)
    # Add labels and legend
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Add mildly visible dotted grid lines in x direction
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.yscale("linear")
    if y_scale == "log":
        plt.yscale("log")
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    # Remove spines from x and y axis
    sns.despine(left=True, bottom=True)

    # Add legend
    plt.legend()

    return plt


def plot_series(time, series, format="-", start=0, end=None):
    """
    Visualizes time series data

    Args:
      time (array of int) - contains the time steps
      series (array of int) - contains the measurements for each time step
      format - line style when plotting the graph
      label - tag for the line
      start - first time step to plot
      end - last time step to plot
    """

    # Setup dimensions of the graph figure
    plt.figure(figsize=(10, 6))

    if type(series) is tuple:
        for series_num in series:
            # Plot the time series data
            plt.plot(time[start:end], series_num[start:end], format)

    else:
        # Plot the time series data
        plt.plot(time[start:end], series[start:end], format)

    # Label the x-axis
    plt.xlabel("Time")

    # Label the y-axis
    plt.ylabel("Value")

    # Overlay a grid on the graph
    plt.grid(True)

    # Draw the graph on screen
    plt.show()


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_kernel_density(
    data_list, labels=None, color_palette=COLORS, title="Kernel Density Plot", x_label="Support", y_label="Estimated Density"
):
    # Set Seaborn style
    sns.set_style(style="white")

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Define a custom color palette if provided
    if color_palette:
        sns.set_palette(sns.color_palette(color_palette))

    # Plot kernel density for each tuple in the list
    for i, (data, label) in enumerate(data_list):
        default_label = f"Dataset {i + 1}" if not labels else None
        sns.kdeplot(
            data,
            fill=True,
            alpha=0.5,
            ax=ax,
            label=label or default_label,
            color=color_palette[i] if color_palette else None,
        )

    plt.title(title, fontsize=28,color="#4C566A", fontweight="bold", loc="left", pad=20)
    # Add labels and legend
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # Remove spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Add legend
    if labels:
        ax.legend()
    # Display the plot
    plt.show()


# Example usage:
# Assuming you have a list of tuples (numpy array, label, color), e.g.,
# data_list = [(np.array([...]), 'Label1', '#ffa322'), (np.array([...]), 'Label2', '#bb1414'), ...]
# Call the function with your data list and labels
# plot_kernel_density(data_list)
