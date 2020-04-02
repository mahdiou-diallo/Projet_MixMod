import scipy.sparse as sps
import scipy.io as sio
from sklearn.preprocessing import LabelEncoder
import argparse


def read_data(file, nrow, ncol, n):
    rows = []
    cols = []
    vals = []
    # first = True
    for r in range(nrow):
        parsed = f.readline().strip().split()
        # if first:
        #     first = False
        #     print(parsed)
        c = map(lambda x: int(x)-1, parsed[::2])
        v = map(float, parsed[1::2])
        rows.extend((r for _ in range(len(parsed)//2)))
        cols.extend(c)
        vals.extend(v)

    print(f"rows: {nrow-1} = {max(rows)}\ncols: {ncol-1} >= {max(cols)}")
    try:
        m = sps.csr_matrix((vals, (rows, cols)),
                           shape=(nrow, ncol), dtype=float)
    except ValueError as e:
        raise e
    return m


parser = argparse.ArgumentParser(
    description="Reads CLUTO files and converts them into a .mat file that can easily be loaded.")
parser.add_argument("-i", "--indir", help="Input folder path", required=True)
parser.add_argument("-n", "--name", help="Dataset name", required=True)
parser.add_argument("-o", "--outdir", help="Output folder path", required=True)
parser.add_argument(
    "-s", "--suffix", help="Suffix to add to the new file", default='_new')

args = parser.parse_args()

inpath = args.indir  # './data/cluto'
name = args.name  # 'cacmcisi'
outpath = args.outdir
suffix = '' if inpath != outpath else args.suffix

with open(f'{inpath}/{name}.mat') as f:
    nrow, ncol, n = map(int, f.readline().strip().split())
    print(f"rows:\t\t{nrow}\ncolumns:\t{ncol}\nnonzero:\t{n}")
    data = read_data(f, nrow, ncol, n)

with open(f'{inpath}/{name}.mat.clabel') as f:
    features = [*map(str.strip, f.readlines())]

with open(f'{inpath}/{name}.mat.rclass') as f:
    labels_n = [*map(str.strip, f.readlines())]
    le = LabelEncoder().fit(labels_n)
    labels = le.transform(labels_n)

sio.savemat(f'{outpath}/{name}{suffix}.mat', {
    'mat': data,
    'fea': features,
    'labels': labels,
    'label_names': le.classes_
})
