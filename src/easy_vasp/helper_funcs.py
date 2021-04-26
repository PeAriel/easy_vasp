import numpy as np
from sys import platform
import os


def get_slash():
    if platform == 'win32':
        return '\\'
    else:
        return '/'


def kpoint_segment(first, second, pps):
    """
    :param first: first k-point
    :param second: second k-point
    :param pps: points per segment
    :return: normalized (by k-space distance) segment for bands plot
    """
    first = np.array(first)
    second = np.array(second)
    dist = np.linalg.norm(second - first)
    segment = np.linspace(0, dist, pps)
    return segment


def frac2cart(frac_point, lat, twopi=False):
    """
    transforms a fractional point to cartesian. assumes that the lattice already has 2pi included
    :param frac_point: fractional coordiante
    :param lat: lattice. can be either real space or reciprocal space. should be a matrix where each row is a
    lattice vector
    :param twopi: determines weather a multiplication by 2pi is needed
    :return: cartesian point
    """
    frac_point = np.array(frac_point)
    lat = np.array(lat)
    for i in range(3):
        lat[i] *= frac_point[i]
    if twopi:
        cart = 2 * np.pi * lat.sum(0)
    else:
        cart = lat.sum(0)
    return cart.tolist()


def smooth(vec, step):
    vec2 = np.zeros(len(vec))
    for i in range(step - 1, len(vec), step):
        vec2[i - (step - 1):i + 1] = vec[i - (step - 1):i + 1].sum() / step
    return vec2
