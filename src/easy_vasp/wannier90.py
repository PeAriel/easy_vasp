import re
import os
from time import time

import numpy as np
from numpy import pi, exp
from src.easy_vasp.helper_funcs import *
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from copy import deepcopy
import matplotlib.colors as colors


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


def _print_xyz_error():
    print("Could not find wannier90_centers.xyz file!!!!")
    print("Continuing calculating the Fourier transform of the principle layers without\n"
          "inter layer couplings. This can result in a less accurate fitting. To improve the fitting,\n"
          "produce a wannier90_centers.xyz.")


def _get_wan_nkpts(path):
    """
    :param path: path to the wannier90.win file
    :return: the number of k-points from the wannier90.win file
    """
    with open(path, 'r') as f:
        line_num = 0
        for line in f:
            if line == '  \n':
                nkpts = line_num
                break
            line_num += 1
    return nkpts


def _get_num_wann(path):
    """
    :param path: path to the wannier90.win file
    :return: the number of wannier bands from the wannier90.win file
    """
    win_path = os.path.dirname(path) + get_slash() + 'wannier90.win'
    num_wann_expr = '\s*num_wann\s=\s(\d+)'
    num_wann = None
    with open(win_path, 'r') as f:
        for line in f:
            match = re.match(num_wann_expr, line)
            if match:
                num_wann = int(match.group(1))
                return num_wann
    if num_wann is None:
        print("\nCould not extract number of wannier bands from wannier90.win file.\n"
              "You can input the number here:")
        return int(input())


def _get_unitcell_cart(path):
    win_path = os.path.dirname(path) + get_slash() + 'wannier90.win'
    unit_cell_expr = 'begin unit_cell_cart'
    unitcell_cart = []
    with open(win_path, 'r') as f:
        for line_num, line in enumerate(f):
            match_unitcell = re.match(unit_cell_expr, line)
            if match_unitcell:
                unitcell_line = line_num
            try:
                if unitcell_line < line_num <= unitcell_line + 3:
                    unitcell_cart += [[float(i) for i in line.split()]]
            except UnboundLocalError:
                pass
    return unitcell_cart


def get_kpath(path):
    win_path = os.path.dirname(path) + get_slash() + 'wannier90.win'
    begin_kpath_regex = 'begin kpoint_path'
    end_kpath_regex = 'end kpoint_path'
    kpath = []
    path_line = 2e5
    with open(win_path, 'r') as f:
        for line_num, line in enumerate(f):
            match_begin_kpath = re.match(begin_kpath_regex, line)
            match_end_kpath = re.match(end_kpath_regex, line)
            if match_begin_kpath:
                path_line = line_num
            if match_end_kpath:
                path_line = 2e5
            if line_num > path_line:
                kpath += [line.split()]
    for i in range(len(kpath)):
        for j in range(3):
            kpath[i][j+1] = float(kpath[i][j+1])
            kpath[i][j + 5] = float(kpath[i][j + 5])
    return kpath


def get_recip_lattice(lattice):
    """
    :param lattice: real space lattice as a matrix where each row is a basis lattice vector
    :return: reciprocal lattice vectors. recip[i] = \vec{b_i}
    """
    lattice = lattice
    vol = np.dot(lattice[0], np.cross(lattice[1], lattice[2]))
    recip = [[], [], []]

    recip[0] = 2 * pi * (np.cross(lattice[1], lattice[2])) / vol
    recip[1] = 2 * pi * (np.cross(lattice[2], lattice[0])) / vol
    recip[2] = 2 * pi * (np.cross(lattice[0], lattice[1])) / vol

    for i in range(3):
        recip[i] = recip[i].tolist()
    return recip


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


class Wannier90Centers:
    def __init__(self, file_path):
        self.file_path = file_path
        self.centers = []
        self._parse()
        self.ncenters = len(self.centers)

    def _parse(self):
        with open(self.file_path, 'r') as f:
            for line in f:
                tmp = line.split()
                if tmp[0] == 'X':
                    tmp.pop(0)
                    self.centers += [[float(j) for j in tmp]]

    def get_phase_matrix(self, k):
        """
        :param k: k point vector at which we evaluate the phases
        :return: exp( 1j*k \cdot (r_n - r_m) )
        """
        phase_mat = np.zeros((self.ncenters, self.ncenters), dtype=complex)
        for i in range(self.ncenters):
            for j in range(i + 1, self.ncenters):
                phase_mat[i, j] = np.exp(1j * (np.dot(k, self.centers[i]) - np.dot(k, self.centers[j])))

        phase_mat += phase_mat.conj().T
        for i in range(self.ncenters):
            phase_mat[i, i] = 1

        return phase_mat


class Wannier90Bands:
    def __init__(self, file_path):
        self.file_path = file_path
        self.num_wann = _get_num_wann(file_path)
        self.nkpts = _get_wan_nkpts(file_path)
        self.unit_cell_cart = _get_unitcell_cart(file_path)
        self.band_energy_vectors = [[0 for _ in range(self.nkpts)] for _ in range(self.num_wann)]
        self.kpoint_vector = [0 for _ in range(self.nkpts)]
        self._parse()

    def _parse(self):
        with open(self.file_path, 'r') as f:
            line_num = 0
            band_num = 0
            for line in f:
                if line == '  \n':
                    line_num += 1
                    band_num += 1
                    continue
                tmp = line.split()
                en = float(tmp[1])
                if band_num == 0:
                    self.kpoint_vector[line_num] = float(tmp[0])
                self.band_energy_vectors[band_num][line_num % (self.nkpts + 1)] = en
                line_num += 1
                if line_num == self.num_wann * (self.nkpts + 1) - 1:
                    break


class Wannier90Hr:

    def __init__(self, file_path):
        self.file_path = file_path
        self.unit_cell_cart = _get_unitcell_cart(file_path)
        self.recip_lattice = get_recip_lattice(self.unit_cell_cart)
        self.num_wan = None
        self.num_sites = None
        self.multiplicity_lines = None
        self.cutoof_line = None
        self.multiplicity_vec = []
        # self.real_hamiltonian is the main outcome of this class; it is a dictionary in which the key is the
        # coordinate (string) e.g '1 -2 2' and the value is the hamiltonian of this real
        # space point (i.e orbital matrix elements)
        self.real_hamiltonian = {}
        # self.minmax_sites gives the min and max fractional coordinate of each site: e.g [[-3, 3], [-1, 2], [0, 5]]
        self.minmax_sites = [[1000, -1000] for _ in range(3)]
        self.kpath = get_kpath(file_path)
        self.kpoint_vector = np.array([])
        self.diag_k_vecs = None
        self._get_params()
        start_time = time()
        self.cart_sympoints_indices = None
        self._parse()
        print('run time: %.1f seconds' % (time() - start_time))
        print('----------------------------------\n')

    def _get_params(self):
        """
        get parameters from the wannier_hr file in order to parse it properly
        :return: attributes to be used later in the _parse() method
        """
        with open(self.file_path, 'r') as f:
            line_num = 0
            for line in f:
                if line_num == 1:
                    self.num_wan = int(line.split()[0])
                if line_num == 2:
                    self.num_sites = int(line.split()[0])
                    self.multiplicity_lines = self.num_sites // 15 + 1
                    self.cutoof_line = 2 + self.multiplicity_lines
                    break
                line_num += 1

    def _parse(self):
        with open(self.file_path, 'r') as f:
            line_num = 0
            multiplicity_idx = 0
            ham_row = []
            ham = []
            for line in f:
                if 3 <= line_num <= self.cutoof_line:
                        self.multiplicity_vec += [int(i) for i in line.split()]

                if line_num > self.cutoof_line:

                    line_vec = [float(i) for i in line.split()]
                    # note that here we divide the the multiplicity of each matrix element
                    ham_row += [(line_vec[5] + 1j * line_vec[6]) / self.multiplicity_vec[multiplicity_idx]]

                    if line_num % self.num_wan == self.cutoof_line and line_num > self.cutoof_line + 1:
                        ham += [ham_row]
                        ham_row = []

                    if line_num % self.num_wan ** 2 == self.cutoof_line:
                        tmp = line.split()
                        new_coordinate = [None for _ in range(3)]
                        for i in range(3):
                            new_coordinate[i] = str(tmp[i])
                        new_coordinate = ' '.join(new_coordinate)
                        self.real_hamiltonian[new_coordinate] = ham
                        ham = []
                        multiplicity_idx += 1

                line_num += 1
        for coordinate in self.real_hamiltonian.keys():
            tmp = [float(i) for i in coordinate.split()]
            for j in range(3):
                if tmp[j] < self.minmax_sites[j][0]:
                    self.minmax_sites[j][0] = tmp[j]
                if tmp[j] > self.minmax_sites[j][1]:
                    self.minmax_sites[j][1] = tmp[j]
        print('\n----------------------------------')
        print('done parsing wannier90_hr.dat file')

    def diagonalize(self, sites='all', pps=50, kpath=None):
        """
        k-space diagonalization of the real Hamiltonain using Fourier transform
        :param sites: real space sites to include. by default it transforms all sites. Otherwise, specify
        a list of integer coordinates
        :param pps: points per segment in reciprocal space. default is 50
        :param kpath: enter the k-path manually. input a list of fractional coordinates like in vasp KPOINTS file.
        e.g: [[M fracs], [G fracs], [G fracs], [K fracs], [K fracs], [A fracs]]
        :return: eigenvalues for each k in wannier.in file (from kpath)
        TODO: create a k-points vector for plot (normalized by the distance)
        TODO: Handle the case where there is no XYZ file in the folder. Ive already written a print function to warn
              the user.
        """
        wannier_centers = Wannier90Centers(os.path.dirname(self.file_path) + get_slash() + 'wannier90_centres.xyz')
        # first build the kpath in cartesian coordinates. either default by wannier90.win or input manually
        if kpath:
            nsegments = len(kpath) // 2
            kvecs = np.zeros(pps * nsegments * 3).reshape(pps * nsegments, 3)
            for i in range(nsegments):
                start_vec = np.array(frac2cart(kpath[2*i], self.recip_lattice))
                end_vec = np.array(frac2cart(kpath[2*i + 1], self.recip_lattice))
                for j in range(3):
                    kvecs[pps * i:pps * (i + 1), j] = np.linspace(start_vec[j], end_vec[j], pps)

            # build a plotable k-path vector
            cart_kpath = np.zeros(nsegments * 2 * 3).reshape(nsegments * 2, 3)
            for j, i in enumerate(range(0, nsegments * 2, 2)):
                cart_kpath[i] = np.array(frac2cart(kpath[2 * j], self.recip_lattice))
                cart_kpath[i + 1] = np.array(frac2cart(kpath[2 * j + 1], self.recip_lattice))

        else:
            nsegments = len(self.kpath)
            kvecs = np.zeros(pps * nsegments * 3).reshape(pps * nsegments, 3)
            for i in range(nsegments):
                start_vec = np.array(frac2cart(self.kpath[i][1:4], self.recip_lattice))
                end_vec = np.array(frac2cart(self.kpath[i][5:8], self.recip_lattice))
                for j in range(3):
                    kvecs[pps * i:pps * (i + 1), j] = np.linspace(start_vec[j], end_vec[j], pps)

            # build a plotable k-path vector
            cart_kpath = np.zeros(nsegments * 2 * 3).reshape(nsegments * 2, 3)
            for j, i in enumerate(range(0, nsegments * 2, 2)):
                cart_kpath[i] = np.array(frac2cart(self.kpath[j][1:4], self.recip_lattice))
                cart_kpath[i + 1] = np.array(frac2cart(self.kpath[j][5:8], self.recip_lattice))

        self.cart_sympoints_indices = [0] + [(n + 1) * pps - 1 for n in range(nsegments)]

        tmp = np.zeros(pps)
        for i in range(0, nsegments):
            last_tmp = tmp[-1]
            tmp = kpoint_segment(cart_kpath[2 * i], cart_kpath[2 * i + 1], pps) + last_tmp
            self.kpoint_vector = np.append(self.kpoint_vector, tmp)

        # initialize arrays for eigenvalues at each k
        dim = len(self.real_hamiltonian[list(self.real_hamiltonian.keys())[0]])
        eigs = np.zeros(dim * len(kvecs)).reshape(len(kvecs), dim)
        eigvecs = np.zeros((len(kvecs), dim, dim), dtype=complex)
        # diagonalize using all coordinates
        if sites == ('all' or 'All'):
            for i, k in enumerate(kvecs):
                k_ham = np.zeros((dim, dim), dtype=complex)
                phase_mat = wannier_centers.get_phase_matrix(k)
                for int_coordinates, real_ham in self.real_hamiltonian.items():
                    lat_vec = ints2vec(self.unit_cell_cart, [int(n) for n in int_coordinates.split()])
                    k_ham += np.array(np.multiply(real_ham, phase_mat)) * exp(1j * np.dot(k, lat_vec))
                eigs[i], eigvecs[i] = np.linalg.eigh(k_ham)
        else:
            # the list of coordinates case
            ncoordinates = len(sites)
            str_coord = [[str(i) for i in coord] for coord in sites]
            coordinates = [' '.join(coord) for coord in str_coord]

            for i, k in enumerate(kvecs):
                k_ham = np.zeros(dim ** 2, dtype=complex).reshape(dim, dim)
                phase_mat = wannier_centers.get_phase_matrix(k)
                for s, coordinate in enumerate(coordinates):
                    lat_vec = ints2vec(self.unit_cell_cart, sites[s])
                    k_ham += np.array(np.multiply(self.real_hamiltonian[coordinate], phase_mat)) * exp(1j * np.dot(k, lat_vec))
                eigs[i], eigvecs[i] = np.linalg.eigh(k_ham)

        self.diag_k_vecs = kvecs
        return eigs.T, eigvecs

    def get_real_inversion_operator(self, projections, spin=True):
        """
        get the real space inversion operator.
        :param projections: a set of projections by the order requested in wannier.win file.
         For example: for 3 d orbitals and 5 p orbitals, input: ['d'] * 3 + ['p'] * 5. important! no need to multiply
         by 2 for spinors, the spin tag takes that into account.
        :param spin: boolean. takes spinors into account (by default set to true)

        p1 was added to account for different p orbital (e.g in Bi2Se3). More orbitals can be added easily
        """
        basis = []
        indicator = [[2, 1], [3, -1], [5, -1], [7, -1], [11, 1], [13, 1], [17, 1], [19, 1], [23, 1],
                     [29, -1], [31, -1], [37, -1]]
        for i, orb in enumerate(projections):
            if 's' in orb:
                basis.append(2)
            elif 'p' in orb:
                basis.append(3)
                basis.append(5)
                basis.append(7)
            elif 'd' in orb:
                basis.append(11)
                basis.append(13)
                basis.append(17)
                basis.append(19)
                basis.append(23)
            elif 'p1' in orb:
                basis.append(29)
                basis.append(31)
                basis.append(37)

        basis = np.array([basis])
        basis = basis.T@basis
        for i in range(9):
            basis[np.where(basis == indicator[i][0] ** 2)] = indicator[i][1]
            for j in range(9):
                basis[np.where(basis == indicator[i][0] * indicator[j][0])] = 0

        wannier_centers = Wannier90Centers(os.path.dirname(self.file_path) + get_slash() + 'wannier90_centres.xyz')
        frac_centers_tmp = cart2frac(wannier_centers.centers, self.unit_cell_cart)
        ncenters = wannier_centers.ncenters
        frac_centers = np.array([[round(i, 3) for i in frac_centers_tmp[j]] for j in range(ncenters)])
        tol = 1e-3

        if spin:
            inversion_op_up = np.zeros((ncenters // 2, ncenters // 2), dtype=int)
            inversion_op_down = np.zeros((ncenters // 2, ncenters // 2), dtype=int)
            for iup in range(ncenters // 2):
                idn = iup + (ncenters // 2)
                for jup in range(ncenters // 2):
                    jdn = jup + (ncenters // 2)
                    if np.linalg.norm((-frac_centers[iup] + 1) % 1 - frac_centers[jup]) < tol:
                        inversion_op_up[iup, jup] = 1
                    if np.linalg.norm((-frac_centers[idn] + 1) % 1 - frac_centers[jdn]) < tol:
                        inversion_op_down[iup, jup] = 1  # the indices here are indeed OK!
            inversion_op_up = np.multiply(inversion_op_up, basis)
            inversion_op_down = np.multiply(inversion_op_down, basis)
            return np.kron([[1, 0], [0, 0]], inversion_op_up) + np.kron([[0, 0], [0, 1]], inversion_op_down)

        else:
            inversion_op = np.zeros((ncenters, ncenters), dtype=int)
            for i in range(ncenters):
                for j in range(ncenters):
                    if np.linalg.norm((-frac_centers[i] + 1) % 1 - frac_centers[j]) < tol:
                        inversion_op[i, j] = 1

            return np.multiply(inversion_op, basis)

    def get_k_inversion_operator(self, projections, kpoint, is_cart=None):
        """
        get the k-space inversion operator. same description as for he real space except for the kpoint tag.
        :param kpoint : list, or nested list of kpoints in fractional coordinate (by default) or cartesian
        (but remember to switch the is_cart tag on!). e.g kpoint=[0.5, 0.5, 0.5] or kpoint=[[0.5, 0, 0], [0, 0, 0]].
        if a list of k-points is given the function returns for each kpoint a matrix:
        (number of kpoints, dim, dim)
        :param is_cart: determines if kpoint is given is fractional (default or cartesian coordinates)
        """
        dim = len(self.real_hamiltonian[list(self.real_hamiltonian.keys())[0]])
        wannier_centers = Wannier90Centers(os.path.dirname(self.file_path) + get_slash() + 'wannier90_centres.xyz')
        real_inversion = self.get_real_inversion_operator(projections)

        is_list = False
        for k in kpoint:
            if type(k) == list:
                is_list = True
                break
            else:
                break

        if is_list:
            parity = np.zeros((len(kpoint), dim, dim), dtype=complex)
            for idx, k in enumerate(kpoint):
                if is_cart is None:
                    k = frac2cart(k, self.recip_lattice)
                k_ham = np.zeros((dim, dim), dtype=complex)
                phase_mat = wannier_centers.get_phase_matrix(k)
                for int_coordinates, real_ham in self.real_hamiltonian.items():
                    lat_vec = ints2vec(self.unit_cell_cart, [int(n) for n in int_coordinates.split()])
                    k_ham += np.array(np.multiply(real_ham, phase_mat)) * exp(1j * np.dot(k, lat_vec))
                _, k_ham_eignvecs = np.linalg.eigh(k_ham)

                for i in range(dim):
                    for j in range(dim):
                        parity[idx, i, j] = np.vdot(k_ham_eignvecs[:, i],
                                               real_inversion @ k_ham_eignvecs[:, j])
        else:
            if is_cart is None:
                kpoint = frac2cart(kpoint, self.recip_lattice)
            k_ham = np.zeros((dim, dim), dtype=complex)
            phase_mat = wannier_centers.get_phase_matrix(kpoint)
            for int_coordinates, real_ham in self.real_hamiltonian.items():
                lat_vec = ints2vec(self.unit_cell_cart, [int(n) for n in int_coordinates.split()])
                k_ham += np.array(np.multiply(real_ham, phase_mat)) * exp(1j * np.dot(kpoint, lat_vec))
            _, k_ham_eignvecs = np.linalg.eigh(k_ham)

            parity = np.zeros((dim, dim), dtype=complex)
            for i in range(dim):
                for j in range(dim):
                    parity[i, j] = np.vdot(k_ham_eignvecs[:, i], real_inversion @ k_ham_eignvecs[:, j])

        return parity
