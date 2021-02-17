import os
from datetime import datetime
from config import ROOT_DIR
from matplotlib import figure


def save_plot(fig: figure, name: str, pgf: str = None):
    """
    Wrapper for the matplotlib save_plot function. Saves all data to the ./plots directory as png and tex files.

    Parameters
    ----------
    fig : matplotlib figure object to be saved
    name : Name of the file
    pgf : pgf/tikz string to write. None if no pgf.
    """
    if not os.path.isdir(f'{ROOT_DIR}/plots'):
        os.mkdir(f'{ROOT_DIR}/plots')
    fig.savefig(f'{ROOT_DIR}/plots/{name}{datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")}.png')
    if pgf:
        # Need to get rid of extra linebreaks. This is important
        pgf = pgf.replace('\r', '')
        with open(f'{ROOT_DIR}/plots/{name}{datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")}.tex', 'w') as tex:
            tex.write(pgf)
