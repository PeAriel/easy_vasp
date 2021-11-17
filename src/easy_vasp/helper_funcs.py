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

def cart2frac(coordinates, lattice):
    lattice = np.array(lattice)
    coordinates = np.array(coordinates)
    if coordinates.shape == (3,):
        return np.linalg.solve(lattice.T, coordinates).tolist()
    else:
        fracs = np.zeros(len(coordinates) * 3).reshape(len(coordinates), 3)
        for i, coordinate in enumerate(coordinates):
            fracs[i] = np.linalg.solve(lattice.T, coordinate)
    return fracs.tolist()

def gauge_fix(vec):
    """
    fixes the gauge of a vector or a list of vectors (returns a numpy array)
    """
    vec = np.array(vec, dtype=complex)
    if type(vec[0]) == np.ndarray:
        for i in range(len(vec)):
            vec[i] = vec[i] * np.exp(-1j * np.angle(vec[i].sum()))
            if vec[i].sum() < 0:
                vec[i] = -vec[i]
    else:
        vec = vec * np.exp(-1j * np.angle(vec.sum()))
        if vec.sum() < 0:
            vec = -vec

    return vec

def ints2vec(unit_cell, integers):
    """
    :param unit_cell:
    :param integers: list of 3 integers [n1, n2, n3] such that R = n1a1 + n2a2 + n3a3
    :return: a lattice vector as a numpy array
    """
    unit_cell = np.array(unit_cell)
    integers = np.array(integers)
    r = np.zeros(3)
    for i in range(3):
        r += integers[i] * unit_cell[i]
    return np.array(r)