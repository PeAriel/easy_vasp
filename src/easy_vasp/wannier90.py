import re
import os
from time import time
from copy import deepcopy

import numpy as np
from numpy import pi, exp
from src.easy_vasp.helper_funcs import *
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from copy import deepcopy
import matplotlib.colors as colors

def _real_hamiltonain_alternating_spin_select(real_hamiltonian, specie='up'):
    dim = len(real_hamiltonian.get('0 0 0'))
    spin_ham = {}
    for key, val in real_hamiltonian.items():
        tmp_ham = []
        if specie == 'up':
            for i in range(0, dim, 2):
                tmp = val[i]
                tmp_ham += [tmp[0::2]]
            spin_ham[key] = tmp_ham
        else:
            for i in range(1, dim, 2):
                tmp = val[i]
                tmp_ham += [tmp[1::2]]
            spin_ham[key] = tmp_ham
    return spin_ham


def _real_hamiltonain_block_spin_select(real_hamiltonian, specie='up'):
    """
    cuts the real Hamiltonain obtained from the real_hamiltonain attribute. it returns a dictionary similarly
    """
    dim = len(real_hamiltonian.get('0 0 0')) // 2
    spin_ham = {}
    for key, val in real_hamiltonian.items():
        tmp_ham = []
        for i in range(dim):
            tmp = val[i]
            if specie == 'up':
                tmp_ham += [tmp[0:dim]]
            else:
                tmp_ham += [tmp[dim:(dim * 2)]]
        spin_ham[key] = tmp_ham
    return spin_ham


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

    def spin_ordered_centers(self):
        """
        transform the basis from block of up-down spins to alternating up-down spins
        """
        spin_ordered_centers = np.zeros((self.ncenters, 3), dtype=float)
        for i in range(self.ncenters // 2):
            spin_ordered_centers[2 * i] = self.centers[i]
            spin_ordered_centers[2 * i + 1] = self.centers[i + (self.ncenters // 2)]

        return spin_ordered_centers

    def get_phase_matrix(self, k, spin=None, spin_order=''):
        """
        :param k: k point vector at which we evaluate the phases
        :param alter: specifies if the Hamiltonain is ordered is alternating fashion or block. Takes values -
         'alternating', 'blocks'. if spin is not needed it may be left as the default.
        :return: exp( 1j*k \cdot (r_n - r_m) )
        """
        phase_mat = np.zeros((self.ncenters, self.ncenters), dtype=complex)
        for i in range(self.ncenters):
            for j in range(i + 1, self.ncenters):
                phase_mat[i, j] = np.exp(1j * (np.dot(k, self.centers[i]) - np.dot(k, self.centers[j])))

        phase_mat += phase_mat.conj().T
        for i in range(self.ncenters):
            phase_mat[i, i] = 1

        if spin == 'up':
            phase_mat_up = []
            if 'block' in spin_order:
                for i in range(self.ncenters//2):
                    tmp = phase_mat[i]
                    phase_mat_up += [tmp[0:self.ncenters // 2]]
                return np.array(phase_mat_up)
            if 'alter' in spin_order:
                for i in range(0, self.ncenters, 2):
                    tmp = phase_mat[i]
                    phase_mat_up += [tmp[0::2]]
                return np.array(phase_mat_up)
        if spin == 'down':
            phase_mat_down = []
            if 'block' in spin_order:
                for i in range(self.ncenters):
                    phase_mat_down += [phase_mat[i][(self.ncenters // 2):self.ncenters]]
                return phase_mat_down
            if 'alter' in spin_order:
                for i in range(1, self.ncenters, 2):
                    tmp = phase_mat[i]
                    phase_mat_down += [tmp[1::2]]
                return np.array(phase_mat_down)

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
                    real = round(line_vec[5], 3)
                    imag = round(line_vec[6], 3)
                    ham_row += [(real + 1j * imag) / self.multiplicity_vec[multiplicity_idx]]

                    if line_vec[3] == self.num_wan:
                        ham += [ham_row]
                        ham_row = []

                    if line_vec[3] == line_vec[4] == self.num_wan:
                        tmp = line.split()
                        new_coordinate = [None for _ in range(3)]
                        for i in range(3):
                            new_coordinate[i] = str(tmp[i])
                        new_coordinate = ' '.join(new_coordinate)
                        self.real_hamiltonian[new_coordinate] = ham
                        ham = []
                        multiplicity_idx += 1

                line_num += 1

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

    def get_real_inversion_operator(self, projections, spin=None):
        """
        get the real space inversion operator.
        :param projections: a set of projections by the order requested in wannier.win file.
         For example: for 3 d orbitals and 5 p orbitals, input: ['d'] * 3 + ['p'] * 5. important! no need to multiply
         by 2 for spinors, the spin tag takes that into account.
        :param spin: boolean. takes spinors into account (by default set to true)
        p1 was added to account for different p orbital (e.g in Bi2Se3). More orbitals can be added easily
        """
        basis = []
        for i, orb in enumerate(projections):
            if orb == 's':
                basis.append(['s', 'up', 1])
            elif orb == 'p':
                basis.append(['px', 'up', -1])
                basis.append(['py', 'up', -1])
                basis.append(['pz', 'up', -1])
            elif orb == 'd':
                basis.append(['dxy', 'up', 1])
                basis.append(['dx2y2', 'up', 1])
                basis.append(['dxz', 'up', 1])
                basis.append(['dyz', 'up', 1])
                basis.append(['dz2', 'up', 1])
            elif orb == 'p1':
                basis.append(['p1x', 'up', -1])
                basis.append(['p1y', 'up', -1])
                basis.append(['p1z', 'up', -1])
            elif orb == 'pz':
                basis.append(['pz', 'up', -1])

        wannier_centers = Wannier90Centers(os.path.dirname(self.file_path) + get_slash() + 'wannier90_centres.xyz')
        frac_centers_tmp = cart2frac(wannier_centers.centers, self.unit_cell_cart)
        ncenters = wannier_centers.ncenters
        frac_centers = np.array([[round(i, 3) for i in frac_centers_tmp[j]] for j in range(ncenters)])
        tol = 1e-2

        if spin:

            # This was the procedure to extend the basis for up-down-up-down-...
            # spin_basis = []
            # for i in range(len(basis)):
            #     spin_basis.append(basis[i])
            #     spin_basis.append(deepcopy(basis[i]))
            #     spin_basis[2 * i + 1][1] = 'down'
            # basis = spin_basis


            # This was the procedure to extend the basis for big up-down blocks
            # for i in range(len(basis)):
            #     basis.append(deepcopy(basis[i]))
            # for i in range(len(basis) // 2, len(basis)):
            #     basis[i][1] = 'down'

            # We treat only(!) the upper spin block
            ncenters //= 2
            inversion_op = np.zeros((ncenters, ncenters), dtype=int)
            for i in range(ncenters):
                for j in range(ncenters):
                    if np.linalg.norm((-frac_centers[i] + 1) % 1 - frac_centers[j]) < tol:
                        if basis[i][0] == basis[j][0] and basis[i][1] == basis[j][1]:
                            inversion_op[i, j] = basis[i][2]

            return inversion_op

        else:
            inversion_op = np.zeros((ncenters, ncenters), dtype=int)
            for i in range(ncenters):
                for j in range(ncenters):
                    if np.linalg.norm((-frac_centers[i] + 1) % 1 - frac_centers[j]) < tol:
                        if basis[i][0] == basis[j][0]:
                            inversion_op[i, j] = basis[i][2]

            return inversion_op

    def get_k_inversion_operator(self, projections, kpoint, spin=None, is_cart=None):
        """
        get the k-space inversion operator. same description as for he real space except for the kpoint tag.
        :param kpoint : list, or nested list of kpoints in fractional coordinate (by default) or cartesian
        (but remember to switch the is_cart tag on!). e.g kpoint=[0.5, 0.5, 0.5] or kpoint=[[0.5, 0, 0], [0, 0, 0]].
        if a list of k-points is given the function returns for each kpoint a matrix:
        (number of kpoints, dim, dim)
        :param is_cart: determines if kpoint is given is fractional (default or cartesian coordinates)
        """
        dim = len(self.real_hamiltonian[list(self.real_hamiltonian.keys())[0]])
        if spin:
            dim //= 2
        wannier_centers = Wannier90Centers(os.path.dirname(self.file_path) + get_slash() + 'wannier90_centres.xyz')
        real_inversion = self.get_real_inversion_operator(projections, spin)

        is_list = False
        for k in kpoint:
            if type(k) == list:
                is_list = True
                break
            else:
                break
        # NOT UPDATED YET!!! APPLY HERE ALL THE CHANGES MADE TO A SINGLE POINT!!!!!!!!!!!
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

            return parity

        else:
            if is_cart is None:
                kpoint = frac2cart(kpoint, self.recip_lattice)
            if spin:
                phase_mat = wannier_centers.get_phase_matrix(kpoint, spin, spin_order='block')
                r_ham = _real_hamiltonain_block_spin_select(self.real_hamiltonian, spin)
            else:
                r_ham = self.real_hamiltonian
                phase_mat = wannier_centers.get_phase_matrix(kpoint)
            k_inversion = np.zeros((dim, dim), dtype=complex)
            k_ham = np.zeros((dim, dim), dtype=complex)
            for int_coordinates, real_ham in r_ham.items():
                lat_vec = ints2vec(self.unit_cell_cart, [int(n) for n in int_coordinates.split()])
                k_ham += np.array(np.multiply(real_ham, phase_mat)) * exp(1j * np.dot(kpoint, lat_vec))
                k_inversion += np.array(np.multiply(real_inversion, phase_mat)) * exp(1j * np.dot(kpoint, lat_vec))
            k_eigs, k_ham_eignvecs = np.linalg.eigh(k_ham)

            k_parity_eigs, k_parity_eignvecs = np.linalg.eigh(k_inversion)
            print(k_eigs)
            print(k_parity_eigs)

            # parity_eignvecs = gauge_fix(parity_eignvecs.T).T
            # k_ham_eignvecs = gauge_fix(k_ham_eignvecs.T).T

            overlaps = np.zeros((dim, dim), dtype=complex)
            for i in range(dim):
                for j in range(dim):
                    overlaps[i, j] = np.vdot(k_ham_eignvecs[:, i], k_parity_eignvecs[:, j])

            parity = np.zeros((dim, dim), dtype=complex)
            for i in range(dim):
                for j in range(dim):
                    parity[i, j] = np.vdot(k_ham_eignvecs[:, i], real_inversion @ k_ham_eignvecs[:, j])

        return parity
