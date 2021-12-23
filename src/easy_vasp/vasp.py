import operator
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.easy_vasp.wannier90 import *
from src.easy_vasp.wannier_tools import *
from sys import platform


class Poscar:
    """
    Used for easy manipulation of a POSCAR file.
    Easily change any field in the file and save it as a valid POSCAR file.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.lattice = None
        self.cart_or_frac = None
        self.upper_part = None                  # upper part of the original file to stitch.
        self.middle_part = None                 # middle part of the original file to stitch.
        self.last_part = None                   # last part of the original file to stitch.
        self.last_part = None                   # last part of the original file to stitch.
        self.unit_cell = None                   # unit cell coordinate matrix.
        self.atom_number = None                 # a list with the number of atoms for each species.
        self.atom_name = None                   # a list with the atoms name for each species.
        self.atoms_coordinates = None           # whole coordinates matrix.
        self.new_data = None                    # new data file to save.
        self._parse()

    def _parse(self):
        with open(self.file_path, 'r') as f:
            data = f.read()                                                         # read file.
            data_mat = [row.split() for i, row in enumerate(data.split('\n'))]      # get a matrix of entries.

            self.upper_part = data_mat[0:2]
            self.middle_part = data_mat[5]
            self.last_part = data_mat[7]

            unit_cell = data_mat[2:5]
            self.unit_cell = [[float(elem) for elem in row] for row in unit_cell]

            self.atom_name = data_mat[5]

            atoms = data_mat[6]
            self.atom_number = [int(x) for x in atoms]

            atoms_coordinates = filter(None, data_mat[8::])
            self.atoms_coordinates = [[float(elem) for elem in row[0:3]] for row in atoms_coordinates]

            self.cart_coordinates = [None for _ in range(len(self.atoms_coordinates))]
            for i, frac_point in enumerate(self.atoms_coordinates):
                self.cart_coordinates[i] = frac2cart(frac_point, self.unit_cell)

            if 'd' in data_mat[7][0].lower():
                self.cart_or_frac = 'frac'
            else:
                self.cart_or_frac = 'cart'

    def get_inversion_centers(self):
        coordinates = self.atoms_coordinates
        if self.cart_or_frac == 'cart':
            coordinates = cart2frac(coordinates, self.unit_cell)

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
                        count = 0

        return inversion_centers


class Kpoints:
    def __init__(self, file_path):
        self.file_path = file_path
        self.npoints = None
        self.nsegments = None
        self.pps = None                 # pps --> points per segment
        self.frac_sympoints = None               # fractional coordinates of the high symmetry points
        self.frac_sympoints_indices = None
        self._parse()

    def _parse(self):
        with open(self.file_path, 'r') as f:
            mat = f.read().splitlines()
            mat = list(filter(None, mat))    # clear empty entries at the end of a file
            self.pps = int(mat[1])
            sympoints = [i.split() for i in mat[4:]]
            self.npoints = len(sympoints)
            self.nsegments = int(len(sympoints)/2)
            if len(sympoints) > 3:
                for i in range(len(sympoints)):
                    extra = len(sympoints[i]) - 3
                    for j in range(extra):
                        del sympoints[i][-1]
            self.frac_sympoints = [[float(j) for j in sympoints[i]] for i in range(len(sympoints))]
            self.frac_sympoints = list(filter(None, self.frac_sympoints))
            if 'C' in mat[3].lower():
                poscar_path = os.path.dirname(self.file_path) + get_slash() + 'POSCAR'
                poscar = Poscar(poscar_path)
                self.frac_sympoints = cart2frac(self.frac_sympoints, poscar.unit_cell)

        self.frac_sympoints_indices = [0] + [(i*self.pps - 1) for i in range(1, self.nsegments + 1)]


class Procar:
    def __init__(self, file_path):
        self.file_path = file_path
        self.is_spin_polarized = None
        self.is_collinear = None
        self.number_of_bands = None
        self.number_of_kpoints = None
        self.number_of_ions = None
        self.kpoints_vector = None  # k points vector (arb units). use to plot each E_n(k)
        self.number_of_lines = None
        self.spin_polarized = None
        self._parse()

        os.chdir(os.path.dirname(__file__))
        self.package_path = os.getcwd()

    def _parse(self):

        outcar = Outcar(os.path.dirname(self.file_path) + get_slash() + 'OUTCAR')
        self.spin_polarized = outcar.spin_polarized
        if outcar.collinear == 'T':
            self.is_collinear = True
        else:
            self.is_collinear = False

        with open(self.file_path, 'r') as f:
            line_number = 0  # same indexing as in python (starts from 0)
            for line in f:
                if line_number == 1:
                    line_vec = line.split()
                    self.number_of_kpoints = int(line_vec[3])
                    self.number_of_bands = int(line_vec[7])
                    self.number_of_ions = int(line_vec[11])
                    break
                line_number += 1

        if self.is_collinear:
            self.number_of_lines = ((((self.number_of_ions + 1) * 4 + 4) * self.number_of_bands) + 3) \
                                   * self.number_of_kpoints + 2 + 1
        else:
            self.number_of_lines = ((((self.number_of_ions + 1) + 4) * self.number_of_bands) + 3) \
                                   * self.number_of_kpoints + 2 + 1

    def procar_matrix(self, check=None):
        """
        :return: if not spin polarized - procar file as a 4 dimensional matrix with indices:
                [kpoint, band, ion+1 (last one is the total), orbital+1(last one is total)]
                 if spin polarized     - procar file as a 5 dimensional matrix with indices:
                [kpoint, band, axis+1(last one is total), ion+1 (last one is the total), orbital+1(last one is total)]
        """
        print('\nReading PROCAR')
        with open(self.file_path, 'r') as f:
            raw_procar = f.read().splitlines()
        print('PROCAR was read successfully')

        if self.is_collinear:
            print('Collinear PROCAR. \nStarting to reorder the data')
            k_start = 3
            ion_jump = 1
            band_jump = (self.number_of_ions + 1) * 4 + 4
            k_jump = (band_jump * self.number_of_bands) + 3

            ordered_procar = [[[[[0 for _ in range(10)] for _ in range(self.number_of_ions + 1)] for _ in range(4)]
                               for _ in range(self.number_of_bands)] for _ in range(self.number_of_kpoints)]

            kpoint_idx = 0
            for kpoint in tqdm(range(k_start, self.number_of_kpoints * k_jump, k_jump),
                               desc='k-points processed'):
                band_idx = 0
                band_start = kpoint + 2
                if kpoint > k_start:
                    kpoint_idx += 1
                for band in range(band_start, (self.number_of_bands + band_start) * band_jump, band_jump):
                    ion_start = band + 3
                    ion_idx = 0
                    axis_idx = 0
                    if band != band_start:
                        band_idx += 1
                    if band_idx == self.number_of_bands:
                        break

                    for ion in range(ion_start, (self.number_of_ions + 1) * 4 + ion_start, ion_jump):
                        if check == [kpoint_idx, band_idx, axis_idx, ion_idx % (self.number_of_ions + 1)]:
                            print('\nk-point line   = {}, \t\t k point index = {}'.format(kpoint + 1, kpoint_idx))
                            print('band line      = {}, \t\t band index    = {}'.format(band + 1, band_idx))
                            print('ion line       = {}, \t\t axis index    = {}, \t\t ion index = {}'
                                  .format(ion + 1, axis_idx, ion_idx % (self.number_of_ions + 1)))
                            print('raw procar = {}\n'.format(raw_procar[ion]))

                        tmp = raw_procar[ion].split()
                        del tmp[0]
                        ordered_procar[kpoint_idx][band_idx][axis_idx][ion_idx % (self.number_of_ions + 1)] \
                            = [float(val) for val in tmp]
                        ion_idx += 1
                        if ion_idx % (self.number_of_ions + 1) == 0:
                            axis_idx += 1

            return ordered_procar

        else:
            print('Non-collinear PROCAR. \nStarting to reorder the data')
            k_start = 3
            ion_jump = 1
            band_jump = (self.number_of_ions + 1) + 4
            k_jump = (band_jump * self.number_of_bands) + 3

            ordered_procar = [[[[0 for _ in range(10)] for _ in range(self.number_of_ions + 1)]
                               for _ in range(self.number_of_bands)] for _ in range(self.number_of_kpoints)]

            kpoint_idx = 0
            for kpoint in tqdm(range(k_start, self.number_of_kpoints * k_jump, k_jump),
                               desc='k-points processed'):
                band_idx = 0
                band_start = kpoint + 2
                if kpoint > k_start:
                    kpoint_idx += 1
                for band in range(band_start, (self.number_of_bands + band_start) * band_jump, band_jump):
                    ion_start = band + 3
                    ion_idx = 0
                    if band != band_start:
                        band_idx += 1
                    if band_idx == self.number_of_bands:
                        break

                    for ion in range(ion_start, self.number_of_ions + ion_start + 1, ion_jump):
                        if check == [kpoint_idx, band_idx, ion_idx]:
                            print('\nk-point line = {}, \t\t k point index = {}'.format(kpoint + 1, kpoint_idx))
                            print('band line    = {}, \t\t band index    = {}'.format(band + 1, band_idx))
                            print('ion line     = {}, \t\t ion index     = {}'.format(ion + 1, ion_idx))
                            print('raw procar = {}'.format(raw_procar[ion]))

                        tmp = raw_procar[ion].split()
                        del tmp[0]
                        ordered_procar[kpoint_idx][band_idx][ion_idx] = [float(val) for val in tmp]
                        ion_idx += 1

        return ordered_procar


class Outcar:
    def __init__(self, file_path, skip=None, mbj=False):
        self.file_path = file_path
        self.folder_path = os.path.dirname(file_path)
        self.number_of_bands = None
        self.mbj = mbj
        self.skip_kpoints = skip
        self.spin_polarized = None
        self.collinear = None
        self.number_of_kpoints = None
        self.direct_lattice_vectors = [[0 for _ in range(3)] for j in range(3)]
        self.reciprocal_lattice_vectors = [[0 for _ in range(3)] for j in range(3)]
        self.cart_sympoints_indices = None
        self.fermi_energy = None
        self.band_energy_vectors = []  # row n is E_n(k).
        self._parse()
        if self.skip_kpoints is None:
            self.ticks = [self.kpoints_vector[i] for i in self.cart_sympoints_indices]

        os.chdir(os.path.dirname(__file__))
        self.path = os.getcwd()

    def _parse(self):
        dimensions_expr = '\sDimension of arrays:'
        collinear_expr = '\s*LNONCOLLINEAR'
        efermi_expr = '\sE-fermi\s:\s*(-?\d+.\d+)'
        spin_expr = '\s*ISPIN\s*=\s*(\d+)'
        lattice_expr = '\s*direct lattice vectors'
        with open(self.file_path, 'r') as f:

            # extract parameters: number of bands and k-points, and Ef
            line_num = 1
            dims_line = 0
            lat_line = 0
            lats = False
            for line in f:
                match_dimensions = re.match(dimensions_expr, line)
                if match_dimensions:
                    dims_line = line_num + 1
                if line_num == dims_line:
                    dimensions = line.split()
                    self.number_of_kpoints = int(dimensions[3])
                    if self.skip_kpoints:
                        self.number_of_kpoints -= self.skip_kpoints
                    self.number_of_bands = int(dimensions[14])

                match_fermi = re.match(efermi_expr, line)
                if match_fermi:
                    self.fermi_energy = float(match_fermi.group(1))
                    line_num += 1
                    break

                match_collinear = re.match(collinear_expr, line)
                if match_collinear:
                    tmp = line.split()
                    self.collinear = tmp[2]

                match_lattice = re.match(lattice_expr, line)
                if match_lattice:
                    lat_tmp = []
                    lat_line = line_num
                    lats = True
                if lat_line < line_num < (lat_line + 4) and lats:
                    lat_tmp.append(line.split())
                line_num += 1

                match_spin = re.match(spin_expr, line)
                if match_spin:
                    tmp = int(match_spin.group(1))
                    if tmp in [0, 1, 11]:
                        self.spin_polarized = False
                    elif tmp == 2:
                        self.spin_polarized = True

            # note that now the first line here is the next line of the loop above!
            # Also note that those methods are used instead of just .readlines() to avoid the \n at the end.
            lines = f.read().splitlines()
            del lines[0:2]
            if self.spin_polarized:
                # note that we take only the up spin component!
                # choosing between up or down spin functionality may be incorporated later
                del lines[0:2]
            if self.skip_kpoints:
                del lines[:(3 + self.number_of_bands) * self.skip_kpoints]
            del lines[(3 + self.number_of_bands) * self.number_of_kpoints::]

            for n in range(1, self.number_of_bands + 1):
                nth_line = lines[n + 1::self.number_of_bands + 3]
                if self.mbj:
                    nth_line = lines[n + 2::self.number_of_bands + 3]
                nth_energies = [float(entry.split()[1]) for entry in nth_line]
                self.band_energy_vectors.append(nth_energies)

            for i in range(3):
                for j in range(3):
                    self.direct_lattice_vectors[i][j] = float(lat_tmp[i][j])
                    self.reciprocal_lattice_vectors[i][j] = float(lat_tmp[i][j + 3])

            if self.skip_kpoints is None:
                kpoints_path = os.path.dirname(self.file_path) + get_slash() + 'KPOINTS'
                kpoints = Kpoints(kpoints_path)
                self.cart_sympoints_indices = kpoints.frac_sympoints_indices

                self.cart_sympoints = [[0 for _ in range(3)] for _ in range(kpoints.npoints)]
                for i, sympoint in enumerate(kpoints.frac_sympoints):
                    self.cart_sympoints[i] = frac2cart(sympoint, self.reciprocal_lattice_vectors, twopi=True)

                self.kpoints_vector = np.array([])
                tmp = np.zeros(kpoints.pps)
                for i in range(0, kpoints.nsegments):
                    last_tmp = tmp[-1]
                    tmp = kpoint_segment(self.cart_sympoints[2 * i], self.cart_sympoints[2 * i + 1],
                                         kpoints.pps) + last_tmp
                    self.kpoints_vector = np.append(self.kpoints_vector, tmp)


    def plot_bands(self, n='All',
                   sympoints=None,
                   energy_range=None,
                   wannier_band_path=None,
                   wannier_hr_path=None,
                   slabek_path=None,
                   fermi_shift=None,
                   title=None,
                   axs=None):
        """
        :param n: optional. by default it plots all bands.
        :param sympoints: list of high symmetry point. Currently supports only equally spaced grid.
        :param energy_range: list with min and max energies to plot.
        :param fermi_shift: determines weather of not to shift the fermi energy. Need to specify the value manually
                            list of the number of bands of 'All' to get all bands.
                            for example, to get bands 1,2,3 put n=[1,2,3].
        :return: plot the band structure
        """

        if fermi_shift:
            bands = [[elem - fermi_shift for elem in vec] for vec in self.band_energy_vectors]
            print('\n----------- Fermi energy is shifted to %f -----------' % fermi_shift)
            ef = 0
            if wannier_band_path:
                wannier = Wannier90Bands(wannier_band_path)
                wan_bands = np.array(wannier.band_energy_vectors) - fermi_shift
            if wannier_hr_path:
                wannier = Wannier90Hr(wannier_hr_path)
                wan_bands = wannier.diagonalize() - fermi_shift
            if slabek_path:
                slabek = SlabEk(slabek_path)
                slab_bands = np.array(slabek.band_energy_vectors)   # already shifted by wannier_tools program

        else:
            bands = self.band_energy_vectors
            ef = self.fermi_energy
            if wannier_band_path:
                wannier = Wannier90Bands(wannier_band_path)
                wan_bands = np.array(wannier.band_energy_vectors)
            if wannier_hr_path:
                wannier = Wannier90Hr(wannier_band_path)
                wan_bands = wannier.diagonalize()
            if slabek_path:
                slabek = SlabEk(slabek_path)
                slab_bands = np.array(slabek.band_energy_vectors) + self.fermi_energy

        if axs is None:
            fig, ax = plt.subplots()
        else:
            ax = axs

        efplot = [ef] * len(self.kpoints_vector)
        ax.plot(self.kpoints_vector, efplot, 'k', linestyle='--', linewidth=1)
        if wannier_band_path:
            for i in range(wannier.num_wann):
                ax.plot(wannier.kpoint_vector, wan_bands[i],
                        linestyle='--', linewidth=0.75, color='red', zorder=3)
        if wannier_hr_path:
            for i in range(len(wan_bands)):
                ax.plot(wannier.kpoint_vector, wan_bands[i],
                        linestyle='--', linewidth=0.75, color='red', zorder=3)
        if slabek_path:
            for i in range(slabek.nbands):
                ax.plot(slabek.kpoint_vector, slab_bands[i],
                        linestyle='--', linewidth=0.75, color='red', zorder=3)

        try:
            if n.lower() == 'all':
                for k in range(self.number_of_bands):
                    ax.plot(self.kpoints_vector, bands[k], 'b', linewidth=1.75)
        except AttributeError:
            for k in n:
                ax.plot(self.kpoints_vector, bands[int(k) - 1], 'b', linewidth=1.75)

        min_energy, max_energy = ax.get_ylim()
        ax.vlines(self.ticks, min_energy, max_energy, 'k')
        ax.set_ylim(min_energy, max_energy)
        ax.set_xticks(self.ticks)

        if sympoints:
            ax.set_xticklabels(sympoints, fontsize=16)
        else:
            ax.set_xticklabels([])

        if energy_range:
            ax.set_ylim(energy_range[0], energy_range[1])

        ax.set_ylabel('Energy (eV)', fontsize=16)
        ax.set_xlim(0, max(self.kpoints_vector))
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2)

        if title:
            ax.set_title(title, fontsize=16)

        if axs is None:
            fig.set_size_inches(7, 6)
            return fig, ax
        # plt.show()
        return ax



class ScfOutcar:
    def __init__(self, file_path):
        self.file_path = file_path
        self.fermi_energy = None
        self._parse()

    def _parse(self):
        fermi_energy_expr = '\s*E-fermi'
        with open(self.file_path, 'r') as f:
            for line in f:
                fermi_match = re.match(fermi_energy_expr, line)
                if fermi_match:
                    tmp = line.split()
                    self.fermi_energy = float(tmp[2])