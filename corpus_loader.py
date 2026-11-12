"""Memory-mapped access to the training corpus (``train.npz``, unzipped into a folder).

    import corpus_loader
    corpus = corpus_loader.load("corpus")         # folder holding train.npz
    frames, actions = corpus.episode(0)            # uint8 [T, 64, 64, 3], float32 [T, 2]

``np.load(...)["frames"]`` on an npz reads the whole array into memory; the files inside the (uncompressed) npz are
mapped instead, so opening the corpus is instant and only the frames you touch are read.
"""

import os
import zipfile

import numpy as np


def _mmap_npz(path):
    """Memory-map every array of an *uncompressed* npz."""
    out = {}
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            if info.compress_type != zipfile.ZIP_STORED:
                raise ValueError(f"{info.filename} in {path} is compressed; it cannot be memory-mapped")
            with open(path, "rb") as f:
                f.seek(info.header_offset)
                header = f.read(30)
                name_len = int.from_bytes(header[26:28], "little")
                extra_len = int.from_bytes(header[28:30], "little")
                start = info.header_offset + 30 + name_len + extra_len
                f.seek(start)
                version = np.lib.format.read_magic(f)
                read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, fortran, dtype = read(f)
                offset = f.tell()
            out[info.filename[:-4]] = np.memmap(path, dtype=dtype, mode="r", shape=shape, offset=offset,
                                                order="F" if fortran else "C")
    return out


class Corpus:
    def __init__(self, root):
        path = os.path.join(root, "train.npz") if os.path.isdir(root) else root
        a = _mmap_npz(path)
        self.frames, self.actions, self.offsets = a["frames"], a["actions"], np.asarray(a["offsets"])
        self.lengths = np.diff(self.offsets)

    def __len__(self):
        return len(self.lengths)

    def episode(self, i):
        s, e = self.offsets[i], self.offsets[i + 1]
        return np.asarray(self.frames[s:e]), np.asarray(self.actions[s:e])


def load(root):
    return Corpus(root)
