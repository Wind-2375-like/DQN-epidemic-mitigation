"""Visualization module"""

import io
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.patches     import Patch
from matplotlib.figure      import Figure
from matplotlib.axes        import Axes
from typing                 import List, Dict, Any

matplotlib.use('Agg')


DPI = 80


def fig_2_numpy(fig:Figure):
    """Convers a matplotlib figure to a numpy array (for .jpg/.png export).

    Args:
        fig (Figure): the figure to export.

    Returns:
        NDArray(uint8): the image-containing array
    """
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw')
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    return img_arr


def plot_time_boolean(dataframe: pd.DataFrame, ax: Axes, true_label:str, false_label:str):
    """Plots a vizualization of an action history (a N_CITIESxN_WEEKS boolean array) in a given matplotlib axis.

    Args:
        dataframe (pd.DataFrame): Dataframe containing the data
        ax (Axes): matplotlib axis to plot to
        true_label (str): label given to the true values
        false_label (str): label given to the false values
    """
    b_array = np.array(dataframe).transpose()*(1.0)
    ax.imshow(b_array, aspect='auto', interpolation=None,
              cmap='bwr', vmin=0, vmax=1)

    legend_elements = [Patch(facecolor='blue', label=false_label),
                       Patch(facecolor='red', label=true_label), ]

    ax.legend(handles=legend_elements)

    y_label_list = dataframe.columns
    ax.set_yticks(list(range(b_array.shape[0])), y_label_list)


class Visualize():

    @staticmethod
    def render_episode_country(info_hist: List[Dict[str,Any]]):
        """Renders a summary vizualization of an epidemic episode across the entire country.

        Args:
            info_hist (List): The history of one epidemic episode.

        Returns:
            NDArray(uint8): the plot as an ndarray-image
        """

        plt.close('all')

        infection_hist = pd.DataFrame(
            [e['parameters']['total'] for e in info_hist[0:-1]])
        confinement_hist = pd.DataFrame(
            [e['action']['confinement'] for e in info_hist[0:-1]])
        isolation_hist = pd.DataFrame(
            [e['action']['isolation'] for e in info_hist[0:-1]])
        hospital_hist = pd.DataFrame(
            [e['action']['hospital'] for e in info_hist[0:-1]])
        vaccinate_hist = pd.DataFrame(
            [e['action']['vaccinate'] for e in info_hist[0:-1]])

        # Firrst plot shows infections/deaths
        fig, ax = plt.subplots(3, 2, figsize=(9, 9))
        infection_hist.plot(y='infected', use_index=True, ax=ax[0, 0])
        infection_hist.plot(y='dead', x='day', ax=ax[0, 0])

        # Second plot shows full state
        infection_hist.plot(y='infected', use_index=True, ax=ax[0, 1])
        infection_hist.plot(y='dead', x='day', ax=ax[0, 1])
        infection_hist.plot(y='exposed', x='day', ax=ax[0, 1])
        infection_hist.plot(y='suceptible', x='day', ax=ax[0, 1])
        infection_hist.plot(y='recovered', x='day', ax=ax[0, 1])

        # The next plots show the actions
        plot_time_boolean(confinement_hist,
                          ax[1, 0], 'Confined', 'Not Confined')
        plot_time_boolean(isolation_hist, ax[1, 1], 'Isolated', 'Not Isolated')
        plot_time_boolean(
            hospital_hist, ax[2, 0], 'With additional hospital beds', 'Without additional hospital beds')
        plot_time_boolean(
            vaccinate_hist, ax[2, 1], 'Vaccinate', 'Do not vaccinate')
        fig.tight_layout()
        return fig_2_numpy(fig)

    @staticmethod
    def render_episode_city(info_hist: List[Dict[str,Any]]):
        """Renders a summary vizualization of an epidemic episode in a per-city manner (it plots deaths and infections for each city).

        Args:
            info_hist (List): The history of one epidemic episode.

        Returns:
            NDArray(uint8): the plot as an ndarray-image
        """

        plt.close('all')

        cities = list(info_hist[0]['parameters']['cities'].keys())
        fig, ax = plt.subplots(len(cities), 1, figsize=(9, 9))

        for _id, city in enumerate(cities):
            c_infected = np.array(
                [e['parameters']['cities'][city]['infected'] for e in info_hist[0:-1]])
            c_dead = np.array([e['parameters']['cities'][city]['dead']
                              for e in info_hist[0:-1]])

            ax[_id].plot(c_infected)
            ax[_id].plot(c_dead)
            ax[_id].legend(['infected', 'dead'])
            ax[_id].set_title(city)

        fig.tight_layout()

        return fig_2_numpy(fig)
