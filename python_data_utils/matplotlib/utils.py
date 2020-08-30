# coding: utf-8

"""
    description: Matplotlib utility functions and classes
    author: Suraj Iyer
"""

__all__ = ['fig2data', 'fig2img']

import numpy as np


def fig2data(figure):
    """
    Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it.
    URL: http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure

    :param figure: a matplotlib figure
    :return: a numpy 3D array of RGBA values
    """
    # draw the renderer
    figure.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = figure.canvas.get_width_height()
    buf = np.fromstring(figure.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = figure.roll(buf, 3, axis=2)
    return buf


def fig2img(figure):
    """
    Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    URL: http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure

    :param fig: a matplotlib figure
    :return: a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(figure)
    w, h, d = buf.shape
    import Image
    return Image.fromstring("RGBA", (w, h), buf.tostring())
