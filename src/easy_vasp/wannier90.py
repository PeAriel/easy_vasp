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

S = np.array([
    [[0, 1], [1, 0]],
    [[0, -1j], [1j, 0]],
    [[1, 0], [0, -1]]
])

def get_layer_sites(axis, wannier_hr, coordinate):
    axes = {'x': 0, 'y': 1, 'z': 2}
    ax = axes.get(axis)
    sites = []
    for site in wannier_hr.real_hamiltonian.keys():
        tmp = [int(j) for j in site.split()]
        if tmp[ax] == coordinate:
            sites += [tmp]
    return sites


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
    num_wann_expr = '\s*num_wann\s*=\s*(\d+)'
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

def _get_atoms_cart(path):
    win_path = os.path.dirname(path) + get_slash() + 'wannier90.win'
    atoms_cart_expr = 'begin atoms_cart'
    end_tag = 'end atoms_cart'
    atoms_cart = []
    with open(win_path, 'r') as f:
        for line_num, line in enumerate(f):
            match_atoms_cart = re.match(atoms_cart_expr, line)
            match_end = re.match(end_tag, line)
            if match_atoms_cart:
                atoms_cart_line = line_num
            if match_end:
                break
            try:
                if atoms_cart_line < line_num:
                    atoms_cart += [[float(i) for i in line.split()[1:4]]]
            except UnboundLocalError:
                pass
    return atoms_cart

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


class Wannier90Centers:
    def __init__(self, file_path, inversion_shift=False):
        self.file_path = file_path
        self.folder_path = get_slash().join(file_path.split(get_slash())[:-1])
        self.centers = []
        self.atoms = []
        self.unit_cell_cart = _get_unitcell_cart(self.folder_path + get_slash() + 'wannier90.win')
        self._parse()
        self.ncenters = len(self.centers)
        self.centers_frac = np.array(cart2frac(self.centers, self.unit_cell_cart))
        if inversion_shift:
            inv_center = self.get_inversion_centers()
            shifted_centers = np.zeros([self.ncenters, 3])
            for i in range(self.ncenters):
                shifted_centers[i] = np.around(self.centers_frac[i] - inv_center, 4)
            self.centers_frac = shifted_centers
            for i in range(len(self.centers)):
                self.centers[i] = frac2cart(shifted_centers[i], self.unit_cell_cart)
            print('Wannier centers shifted to inversion center {}'.format(inv_center))

    def _parse(self):
        with open(self.file_path, 'r') as f:
            for line in f:
                tmp = line.split()
                if tmp[0] == 'X':
                    tmp.pop(0)
                    self.centers += [[float(j) for j in tmp]]
                elif len(self.centers) > 0:
                    tmp.pop(0)
                    self.atoms += [[float(j) for j in tmp]]

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

    def get_difference_matrix(self, axis):
        """
        gives the difference of the wannier centers components (axis)
        """
        axis_dict = {'x': 0, 'y': 1, 'z': 2}
        diff_mat = np.zeros((self.ncenters, self.ncenters))
        for i in range(self.ncenters):
            for j in range(i + 1, self.ncenters):
                diff_mat[i, j] = self.centers[i][axis_dict.get(axis)] - self.centers[j][axis_dict.get(axis)]

        return diff_mat - diff_mat.T

    def get_inversion_centers(self):
        coordinates = cart2frac(self.centers, self.unit_cell_cart)

        dim = len(coordinates)

        coordinates = np.array(coordinates)
        centers_mat = []
        for i, R_i in enumerate(coordinates):
            row = []
            for j, R_j in enumerate(coordinates):
                r_i = np.array([round(n, 5) for n in R_i])
                r_j = np.array([round(n, 5) for n in R_j])
                row += [(r_i + r_j) / 2]
                if j == (dim - 1):
                    centers_mat += [row]

        inversion_centers = []
        tol = 1e-3
        for i in range(dim):
            for j in range(i + 1, dim):
                shifted_coordinates = (coordinates + centers_mat[i][j] + 1) % 1
                #inverted_coordinates = (-shifted_coordinates + 1) % 1
                inverted_coordinates = (-shifted_coordinates + 1) % 1
                count = 0
                for inverted_coordinate in inverted_coordinates:
                    for shifted_coordinate in shifted_coordinates:
                        if np.linalg.norm(inverted_coordinate - shifted_coordinate) < tol:
                            count += 1
                            break
                    if count == dim:
                        if centers_mat[i][j].tolist() not in inversion_centers:
                            inversion_centers.append(centers_mat[i][j].tolist())
                            return (np.array(inversion_centers[0]) + 1) % 1
                        # count = 0

        # return inversion_centers[0]


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

    def __init__(self, file_path, inversion_shift=False):
        self.file_path = file_path
        self.unit_cell_cart = _get_unitcell_cart(file_path)
        self.atoms_cart = _get_atoms_cart(file_path)
        self.atoms_frac = cart2frac(self.atoms_cart, self.unit_cell_cart)
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
        self.wannier_centers = Wannier90Centers(os.path.dirname(self.file_path) + get_slash() + 'wannier90_centres.xyz', inversion_shift)
        self.sigmay = np.kron([[0, -1j], [1j, 0]], np.eye(self.num_wan // 2))

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
                    if self.num_sites % 15 == 0:
                        self.multiplicity_lines -= 1
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
                    real = line_vec[5]
                    imag = line_vec[6]
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

    def fix_TR(self):
        inv_isy = np.linalg.inv(1j * self.sigmay)
        for site, real_ham in self.real_hamiltonian.items():
            self.real_hamiltonian[site] = (np.array(real_ham) + (1j * inv_isy @ np.array(real_ham).conj() @ self.sigmay)) / 2

    def fix_inversion(self, projections, spin=None):
        """
        Enforce real space inversion symmetry.
        :param projections: a set of projections by the order requested in wannier.win file.
                            For example: for 3 d orbitals and 5 p orbitals, input: ['d'] * 3 + ['p'] * 5. important! no need to multiply
                            by 2 for spinors, the spin tag takes that into account.
        :param spin:        boolean. takes spinors into account (by default set to true)
                            p1 was added to account for different p orbital (e.g in Bi2Se3). More orbitals can be added easily
        """
        inversion = self.get_real_inversion_operator(projections, spin)  # real inversion operator in orbital basis
        for site, real_ham in self.real_hamiltonian.items():
            int_list_site = np.array([int(i) for i in site.split()])
            inverted_int_list_site = -int_list_site
            inverted_site = ' '.join([str(i) for i in inverted_int_list_site])
            real_ham = np.array(self.real_hamiltonian[site])
            fham = np.zeros([self.num_wan, self.num_wan], dtype=complex)
            for n in range(self.num_wan):
                for m in range(self.num_wan):
                    p1 = np.where(inversion[n] != 0)[0][0]
                    p2 = np.where(inversion[m] != 0)[0][0]
                    fham[n, m] = (real_ham[n,m] + inversion[n, p1]*inversion[m, p2] * self.real_hamiltonian[inverted_site][p1, p2]) / 2

            self.real_hamiltonian[site] = fham

    def fix_mirror(self, axis, spin=True, nonsym=None):
        """
        fixes mirror symmetry in wannier_hr.
        NOTE!! at the moment this method works only for materials with atoms of the same type and up to p orbitals. Also, we assume that
        the spin in polarized in the z direction.
        :param axis: str representing mirror axis. e.g, for Mx input 'x'
        :param projections: same as in the other methods. Check the docs.
        :param spin: bool.
        :param nonsym: list representing the symmorphic part of the transformation (in fractional coordinates). e.g [0.5, 0.5, 0.0]
        """
        axis_dict = {'x': 0, 'y': 1, 'z': 2}
        orb_dict = {'z': 1, 'x': 2, 'y': 3}
        if nonsym:
            nonsym = np.array(nonsym)
        else:
            nonsym = np.array([0, 0, 0])
        if axis == 'z':
            spin_flip = False
        else:
            spin_flip = True

        atoms_frac = np.around(self.atoms_frac, 5)
        proj_per_atom = self.num_wan // len(atoms_frac)
        if spin:
            proj_per_atom //= 2

        mirrored_idxs = np.zeros(len(atoms_frac), dtype=int)
        for j in range(len(atoms_frac)):
            matom = deepcopy(atoms_frac[j])
            matom[axis_dict.get(axis)] *= -1
            matom += nonsym
            if matom[axis_dict.get(axis)] < 0:
                matom[axis_dict.get(axis)] += 1
            for idx, atom in enumerate(atoms_frac):
                if np.linalg.norm(matom % 1 - atom % 1) < 1e-3:
                    mirrored_idxs[j] = (idx - j) * proj_per_atom
                    break
        if spin and spin_flip:
            mirrored_idxs += self.num_wan // 2

        for site, real_ham in self.real_hamiltonian.items():
            int_sites = np.array([int(s) for s in site.split()])
            new_site = int_sites + matom.astype(int)
            new_site[axis_dict.get(axis)] *= -1
            new_site_str = ' '.join([str(s) for s in new_site])
            atom_num = 0
            orb_num = 0
            for n in range(self.num_wan):
                if n % proj_per_atom == 0 and n != 0:
                    atom_num += 1
                    orb_num = 0
                if spin and n == self.num_wan // 2:
                    atom_num = 0
                atom_mum = 0
                orb_mum = 0
                for m in range(self.num_wan):
                    if m % proj_per_atom == 0 and m != 0:
                        atom_mum += 1
                        orb_mum = 0
                    if spin and m == self.num_wan // 2:
                        atom_mum = 0

                    mirrored_n = (n + mirrored_idxs[atom_num]) % self.num_wan
                    mirrored_m = (m + mirrored_idxs[atom_mum]) % self.num_wan
                    # print('n = {}\tm = {}\norb_num = {}\torb_mum = {}\natom_num = {}\tatom_mum = {}\nmirrored_n = {}\tmirrored_m = {}'.format(n, m, orb_num, orb_mum, atom_num, atom_mum, mirrored_n, mirrored_m))
                    if (orb_mum == orb_dict.get(axis) or orb_num == orb_dict.get(axis)) and n != m:
                        new_val = (real_ham[n][m] - self.real_hamiltonian[new_site_str][mirrored_n][mirrored_m]) / 2
                        self.real_hamiltonian[site][n][m] = new_val
                        self.real_hamiltonian[new_site_str][mirrored_n][mirrored_m] = -new_val
                    else:
                        new_val = (real_ham[n][m] + self.real_hamiltonian[new_site_str][mirrored_n][mirrored_m]) / 2
                        self.real_hamiltonian[site][n][m] = new_val
                        self.real_hamiltonian[new_site_str][mirrored_n][mirrored_m] = new_val
                    orb_mum += 1
                orb_num += 1

    def add_strain(self, magnitude):
        for site, real_ham in self.real_hamiltonian.items():
            int_site = [int(i) for i in site.split()]  # convert string like '1 0 0' to integers list --> [1, 0, 0]
            if int_site[0] == int_site[1] and int_site[0] > 0:
                self.real_hamiltonian[site] = np.array(self.real_hamiltonian[site])
                self.real_hamiltonian[site] *= (1 + magnitude)
            elif int_site[0] == int_site[1] and int_site[0] < 0:
                self.real_hamiltonian[site] = np.array(self.real_hamiltonian[site])
                self.real_hamiltonian[site] *= (1 - magnitude)

    def add_2Dstrain_new(self, dir, mag):
        dir = np.array(dir)
        dir = dir / np.linalg.norm(dir)
        frac_centers = np.array(cart2frac(self.wannier_centers.centers, self.unit_cell_cart))
        for key, real_ham in self.real_hamiltonian.items():
            coord = np.array([int(i) for i in key.split()])
            for n in range(self.num_wan):
                for m in range(self.num_wan):
                    rnm = frac_centers[m] - frac_centers[n]
                    self.real_hamiltonian[key][n][m] *= (1 - mag * np.abs(np.dot(dir, rnm + coord)) / np.linalg.norm(rnm + coord + 1e-3))

    def add_2Dstrain_cart(self, dir, mag):
        dir = np.array(dir)
        dir = dir / np.linalg.norm(dir)
        self.wannier_centers.centers = np.array(self.wannier_centers.centers)
        for key, real_ham in self.real_hamiltonian.items():
            coord = np.array(ints2vec(self.unit_cell_cart, [int(i) for i in key.split()]))
            for n in range(self.num_wan):
                for m in range(self.num_wan):
                    rnm = self.wannier_centers.centers[m] - self.wannier_centers.centers[n]
                    self.real_hamiltonian[key][n][m] *= (1 - mag * np.abs(np.dot(dir, rnm + coord)) / np.linalg.norm(rnm + coord + 1e-3))


    def add_strain_exp(self, magnitude):
        n11 = 0
        for site, real_ham in self.real_hamiltonian.items():
            int_site = [int(i) for i in site.split()]  # convert string like '1 0 0' to integers list --> [1, 0, 0]
            if (int_site[0] == int_site[1]) and (int_site[0] > 0) and (int_site[2] == 0):
                n11 += 1

        linear_mag = magnitude / n11
        for site, real_ham in self.real_hamiltonian.items():
            int_site = [int(i) for i in site.split()]  # convert string like '1 0 0' to integers list --> [1, 0, 0]
            if int_site[0] == int_site[1] and int_site[0] > 0:
                self.real_hamiltonian[site] = np.array(self.real_hamiltonian[site])
                self.real_hamiltonian[site] *= (1 + linear_mag * int_site[0])
            elif int_site[0] == int_site[1] and int_site[0] < 0:
                self.real_hamiltonian[site] = np.array(self.real_hamiltonian[site])
                self.real_hamiltonian[site] *= (1 + linear_mag * int_site[0])


    def break_mirror(self, magnitude):
        orb_mat = np.eye(4)
        orb_mat[2, 2] *= -1
        mirror = np.kron(np.kron(S[0], np.eye(8)), orb_mat).real
        for site in self.real_hamiltonian.keys():
            int_site = np.array([int(s) for s in site.split()])
            real_ham = deepcopy(self.real_hamiltonian[site])
            if int_site[0] >= 0 and sum(int_site) != 0:
                mint_site = deepcopy(int_site)
                mint_site[0] *= -1
                msite = ' '.join([str(s) for s in mint_site])
                self.real_hamiltonian[site] = (real_ham + (1 + magnitude)*mirror.T @ self.real_hamiltonian[msite] @ mirror) / 2
                self.real_hamiltonian[msite] = (self.real_hamiltonian[msite] +  (1 - magnitude)*mirror.T @ real_ham @ mirror) / 2


    def add_onsite(self, projections, delta, spin=None):
        inversion = self.get_real_inversion_operator(projections, spin)
        counted_list = []
        for i in range(self.num_wan // 2):
            if i in counted_list:
                continue
            inverted = np.where(inversion[i] != 0)[0][0]
            self.real_hamiltonian['0 0 0'][i][i] += delta
            self.real_hamiltonian['0 0 0'][inverted][inverted] -= delta
            self.real_hamiltonian['0 0 0'][i + self.num_wan // 2][i + self.num_wan // 2] += delta
            self.real_hamiltonian['0 0 0'][inverted + self.num_wan // 2][inverted + self.num_wan // 2] -= delta
            counted_list.append(inverted)


    def diagonalize(self, sites='all', pps=50, kpath=None, slab=None, spin_split=None):
        """
        k-space diagonalization of the real Hamiltonain using Fourier transform
        :param sites: real space sites to include. by default it transforms all sites. Otherwise, specify
        a list of integer coordinates
        :param pps: points per segment in reciprocal space. default is 50
        :param kpath: enter the k-path manually. input a list of fractional coordinates like in vasp KPOINTS file.
        e.g: [[M fracs], [G fracs], [G fracs], [K fracs], [K fracs], [A fracs]]
        :param slab: 2 dim array with the first entry specifying the slab direction and the second, the layer number, e.g.
        slab = ['z', 2]
        TODO: Handle the case where there is no XYZ file in the folder. Ive already written a print function to warn
              the user.
        """
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
                phase_mat = self.wannier_centers.get_phase_matrix(k)
                for int_coordinates, real_ham in self.real_hamiltonian.items():
                    lat_vec = ints2vec(self.unit_cell_cart, [int(n) for n in int_coordinates.split()])
                    k_ham += np.array(np.multiply(real_ham, phase_mat)) * exp(1j * np.dot(k, lat_vec))
                if spin_split:
                    eigs[i], eigvecs[i] = np.linalg.eigh(k_ham + spin_split * np.kron([[1, 0], [0, -1]], np.eye(self.num_wan // 2)))
                else:
                    eigs[i], eigvecs[i] = np.linalg.eigh(k_ham)

        else:
            # the list of coordinates case
            ncoordinates = len(sites)
            str_coord = [[str(i) for i in coord] for coord in sites]
            coordinates = [' '.join(coord) for coord in str_coord]

            for i, k in enumerate(kvecs):
                k_ham = np.zeros(dim ** 2, dtype=complex).reshape(dim, dim)
                phase_mat = self.wannier_centers.get_phase_matrix(k)
                for s, coordinate in enumerate(coordinates):
                    lat_vec = ints2vec(self.unit_cell_cart, sites[s])
                    k_ham += np.array(np.multiply(self.real_hamiltonian[coordinate], phase_mat)) * exp(1j * np.dot(k, lat_vec))
                if spin_split:
                    eigs[i], eigvecs[i] = np.linalg.eigh(
                        k_ham + spin_split * np.kron([[1, 0], [0, -1]], np.eye(self.num_wan // 2)))
                else:
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


        frac_centers_tmp = cart2frac(self.wannier_centers.centers, self.unit_cell_cart)
        ncenters = self.wannier_centers.ncenters
        frac_centers = np.array([[round(i, 3) for i in frac_centers_tmp[j]] for j in range(ncenters)])
        #for i, cen in enumerate(frac_centers):
        #    frac_centers[i] = (frac_centers[i] + 1) % 1
        tol = 5e-3

        if spin:
            for i in range(len(basis)):
                basis.append(deepcopy(basis[i]))
            for i in range(len(basis) // 2, len(basis)):
                basis[i][1] = 'down'

            inversion_op = np.zeros((ncenters, ncenters), dtype=int)
            for i in range(ncenters):
                for j in range(ncenters):
                    #if np.linalg.norm((-frac_centers[i] + 1) % 1 - frac_centers[j]) < tol:
                    if np.linalg.norm(-frac_centers[i] - frac_centers[j]) < tol:
                        if basis[i][0] == basis[j][0] and basis[i][1] == basis[j][1]:
                            inversion_op[i, j] = basis[i][2]

            return inversion_op

        else:
            inversion_op = np.zeros((ncenters, ncenters), dtype=int)
            for i in range(ncenters):
                for j in range(ncenters):
                    if np.linalg.norm(-frac_centers[i] - frac_centers[j]) < tol:
                        if basis[i][0] == basis[j][0]:
                            inversion_op[i, j] = basis[i][2]

            return inversion_op

    def get_k_inversion_eigs(self, projections, kpoint, spin=None, is_cart=None):
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
            phase_mat = wannier_centers.get_phase_matrix(kpoint)
            k_inversion = np.zeros((dim, dim), dtype=complex)
            k_ham = np.zeros((dim, dim), dtype=complex)
            for int_coordinates, real_ham in self.real_hamiltonian.items():
                lat_vec = ints2vec(self.unit_cell_cart, [int(n) for n in int_coordinates.split()])
                k_ham += np.array(np.multiply(real_ham, phase_mat)) * exp(1j * np.dot(kpoint, lat_vec))
                k_inversion += np.array(np.multiply(real_inversion, phase_mat)) * exp(1j * np.dot(kpoint, lat_vec))
            k_eigs, k_ham_eignvecs = np.linalg.eigh(k_ham)

            # k_parity_eigs, k_parity_eignvecs = np.linalg.eigh(k_inversion)
            # print(k_eigs)
            # print(k_parity_eigs)


            parity = np.zeros((dim), dtype=complex)
            for i in range(dim):
                    parity[i] = np.vdot(k_ham_eignvecs[:, i], real_inversion @ k_ham_eignvecs[:, i])

        return parity.real
