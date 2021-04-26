import re
import numpy as np
from src.easy_vasp.helper_funcs import *

class WtIn:
    def __init__(self, file_path):
        self.file_path = file_path
        self.nk1 = None
        self.slab_segments = None
        self.slab_nkpts = None
        self._parse()

    def _parse(self):
        with open(self.file_path, 'r') as f:
            nk1_regex = 'Nk1\s=\s(\d+)'
            slab_segments_regex = 'KPATH_SLAB'
            for line_num, line in enumerate(f):
                nk1_match = re.match(nk1_regex, line)
                slab_segments_match = re.match(slab_segments_regex, line)
                if nk1_match:
                    self.nk1 = int(nk1_match.group(1))
                if slab_segments_match:
                    slab_segments_line = line_num
                try:
                    if line_num == slab_segments_line + 1:
                        self.slab_segments = int(line.split()[0])
                except UnboundLocalError:
                    pass

        self.slab_nkpts = self.slab_segments * self.nk1


class SlabEk:
    def __init__(self, file_path):
        self.file_path = file_path
        wt_in = WtIn(os.path.dirname(file_path) + get_slash() + 'wt.in')
        self.slab_nkpts = wt_in.slab_nkpts
        self.kpoint_vector = None
        self.band_energy_vectors = None  # every row is a band energy vector corresponding to the kpoints vector above
        self.nbands = None
        self._parse()

    def _parse(self):
        with open(self.file_path, 'r') as f:
            dat_mat = f.read().splitlines()
            del dat_mat[0]
            for i, line in enumerate(dat_mat):
                dat_mat[i] = [float(j) for j in line.split()]
            dat_mat = list(filter(None, dat_mat))
            self.nbands = int(len(dat_mat) / self.slab_nkpts)
            dat_mat = np.array(dat_mat)
            self.kpoint_vector = dat_mat[0:self.slab_nkpts, 0].tolist()
            self.band_energy_vectors = [dat_mat[self.slab_nkpts * k:self.slab_nkpts * (k + 1), 1] for k in range(self.nbands)]
            return
