import matplotlib.pyplot as plt
import numpy as np


def scale_axis_tick_labels(ax, scale=1e-3, axis="x", tick_format="{:.0f}"):
    """
    Scales the tick labels of a specified axis (x or y) by a given factor and formats them.

    Parameters:
    ax (matplotlib.axes.Axes): The Matplotlib axis to modify.
    scale (float): The factor by which to scale the axis ticks.
    axis (str): The axis to modify ('x' or 'y').
    tick_format (str): A format string for the tick labels (default is "{:.1f}").

    Returns:
    None
    """
    if axis == "x":
        ticks = ax.get_xticks()
        xlims = ax.get_xlim()
        scaled_ticks = ticks * scale
        formatted_labels = [tick_format.format(tick) for tick in scaled_ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(formatted_labels)
        ax.set_xlim(xlims)  # Ensure limits remain unchanged after scaling
    elif axis == "y":
        ticks = ax.get_yticks()
        ylims = ax.get_ylim()
        scaled_ticks = ticks * scale
        formatted_labels = [tick_format.format(tick) for tick in scaled_ticks]
        ax.set_yticks(ticks)
        ax.set_yticklabels(formatted_labels)
        ax.set_ylim(ylims)  # Ensure limits remain unchanged after scaling
    else:
        raise ValueError("Axis must be 'x' or 'y'.")


def center_axis_ticks(ax, axis="x"):
    """
    Centers the tick labels of a specified axis (x or y) around zero.

    Parameters:
    ax (matplotlib.axes.Axes): The Matplotlib axis to modify.
    axis (str): The axis to modify ('x' or 'y').

    Returns:
    None
    """
    if axis == "x":
        ticks = ax.get_xticks()
        xlims = ax.get_xlim()
        centered_ticks = ticks + (ticks[1] - ticks[0]) / 2
        ax.set_xticks(centered_ticks)
        ax.set_xticklabels([f"{tick:.0f}" for tick in ticks])
        ax.set_xlim(xlims)  # Ensure limits remain unchanged after centering
    elif axis == "y":
        ticks = ax.get_yticks()
        ylims = ax.get_ylim()
        centered_ticks = ticks + (ticks[1] - ticks[0]) / 2
        ax.set_yticks(centered_ticks)
        ax.set_yticklabels([f"{tick:.0f}" for tick in ticks])
        ax.set_ylim(ylims)  # Ensure limits remain unchanged after centering
    else:
        raise ValueError("Axis must be 'x' or 'y'.")


def evaluate(y_true, y_pred):
    """
    Computes the accuracy of predictions compared to true labels.

    Parameters:
    y_true (array-like): The true labels.
    y_pred (array-like): The predicted labels.

    Returns:
    float: The accuracy as a value between 0 and 1.
    """
    true_assignment = y_true["assignment"]
    predicted_assignment = y_pred["assignment"]
    evaluate_assignment = np.mean(
        np.all(true_assignment == predicted_assignment, axis=[1, 2])
    )

    regression_true = y_true["regression"]
    regression_pred = y_pred["regression"]
    evaluate_regression = np.mean(
        np.square(regression_true - regression_pred).sum(axis=[1, 2])
    )
    return {
        "assignment": evaluate_assignment,
        "regression": evaluate_regression,
    }
