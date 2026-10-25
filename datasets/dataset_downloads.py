'''Best-effort auto-download of the raw dataset files import_dataset() expects on disk.

import_dataset() calls ensure_dataset_files(dataset_name, ...) once, before its big
if/elif dispatch, so the per-dataset download logic doesn't have to live inside that
function -- each dataset's download case lives here instead.

Confidence varies a lot by dataset, since these come from a lot of different places:
  - squirrel, cora-ml, citeseer: direct HTTP download from known public raw-file URLs.
    Should just work.
  - texas / wisconsin / cornell (WebKB), disney / enron / books / amazon (pygod): the
    underlying libraries (torch_geometric.datasets.WebKB, pygod.utils.load_data) already
    auto-download on first use -- nothing to do here. Kept as explicit no-ops below so
    every dataset name import_dataset() knows about is accounted for in one place.
  - JohnsHopkins55 (facebook100), photo / reddit / elliptic.mat (GGAD_datasets): best-effort
    from the sources these were originally collected from. I couldn't verify these URLs are
    still live from this environment -- if the download fails, or 404s, the raised error
    says where to get the file by hand and exactly what path to put it at.
  - elliptic (directed, Kaggle-sourced): requires the `kaggle` pip package and API
    credentials (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY env vars). Downloads +
    unzips the raw CSVs, then runs the existing clean_raw.py to build the .pt file
    import_dataset() actually reads.

None of this touches OGB (PygLinkPropPredDataset already auto-downloads), the simulated/
synthetic dataset names (simulate_dataset generates them, nothing to fetch), or a
caller-supplied `data=` object.
'''
import os
import subprocess
import urllib.request

from utils.printing_utils import printd

DATASETS_DIR = os.path.dirname(os.path.abspath(__file__))


def ensure_dataset_files(dataset_name, to_undirected=False):
    '''Call before loading dataset_name. Downloads whatever raw files that dataset needs, if
    they aren't already on disk. Safe no-op for any name not handled here (synthetic datasets,
    OGB, or datasets whose loader library already auto-downloads).'''
    if dataset_name == 'elliptic':
        # 'elliptic' means two different files depending on to_undirected -- see import_dataset().
        if to_undirected:
            _download_ggad('elliptic')
        else:
            _download_elliptic_directed()
        return
    handler = _DOWNLOADERS.get(dataset_name)
    if handler is not None:
        handler()


def _download(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    printd(f'dataset_downloads: fetching {url} -> {dest_path}')
    tmp_path = dest_path + '.part'
    try:
        urllib.request.urlretrieve(url, tmp_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    os.replace(tmp_path, dest_path)


def _download_squirrel():
    base = os.path.join(DATASETS_DIR, 'wikipedia', 'squirrel')
    root_url = 'https://raw.githubusercontent.com/graphdml-uiuc-2019/geom-gcn/master/new_data/squirrel'
    files = ['musae_squirrel_edges.csv', 'musae_squirrel_features.json', 'musae_squirrel_target.csv']
    for fname in files:
        dest = os.path.join(base, fname)
        if not os.path.exists(dest):
            try:
                _download(f'{root_url}/{fname}', dest)
            except Exception as e:
                raise RuntimeError(
                    f"couldn't auto-download {fname} from {root_url}/{fname} ({e}). "
                    f"This is the geom-gcn mirror of the MUSAE squirrel dataset -- if that "
                    f"repo/path has moved, find musae_squirrel_{{edges.csv,features.json,"
                    f"target.csv}} and place them at {base}/"
                )


def _download_planetoid_npz(name):
    dest = os.path.join(DATASETS_DIR, 'Planetoid', name, f'{name}.npz')
    if os.path.exists(dest):
        return
    url = f'https://github.com/klicperajo/ppnp/raw/master/ppnp/data/{name}.npz'
    try:
        _download(url, dest)
    except Exception as e:
        raise RuntimeError(
            f"couldn't auto-download {name}.npz from {url} ({e}). This is the sparsegraph "
            f"npz format used by several GCN papers (klicperajo/ppnp repo) -- if that's moved, "
            f"find {name}.npz in that format and place it at {dest}."
        )


def _download_facebook100():
    dest = os.path.join(DATASETS_DIR, 'facebook100', 'JohnsHopkins55.mat')
    if os.path.exists(dest):
        return
    url = 'https://escience.rpi.edu/data/DA/fb100/JohnsHopkins55.mat'
    try:
        _download(url, dest)
    except Exception as e:
        raise RuntimeError(
            f"couldn't auto-download JohnsHopkins55.mat from {url} ({e}). This is the "
            f"Facebook100 dataset (Traud/Mucha/Porter) -- if that source has moved, get "
            f"JohnsHopkins55.mat from wherever Facebook100 is currently hosted and place it "
            f"at {dest}."
        )


def _download_ggad(name):
    dest = os.path.join(DATASETS_DIR, 'anomaly_detection', 'GGAD_datasets', f'{name}.mat')
    if os.path.exists(dest):
        return
    # I don't have a verified, scriptable direct-download URL for these -- they're commonly
    # distributed through the GGAD benchmark repo / Google Drive links rather than a stable
    # HTTP path, so failing loudly here beats guessing a URL that silently 404s or grabs the
    # wrong file.
    raise RuntimeError(
        f"{name}.mat is missing at {dest} and I don't have a verified direct-download URL for "
        f"the GGAD_datasets .mat files. Get it from the GGAD benchmark "
        f"(https://github.com/mala-lab/GGAD) or wherever this copy was originally sourced, "
        f"and place it at that exact path."
    )


def _download_elliptic_directed():
    root = os.path.join(DATASETS_DIR, 'anomaly_detection', 'directed', 'elliptic-data-set', 'elliptic_bitcoin_dataset')
    pyg_path = os.path.join(root, 'elliptic_labeled_directed_pyg.pt')
    if os.path.exists(pyg_path):
        return
    features_path = os.path.join(root, 'elliptic_txs_features.csv')
    if not os.path.exists(features_path):
        if subprocess.run(['which', 'kaggle'], capture_output=True).returncode != 0:
            raise RuntimeError(
                "the elliptic dataset's raw CSVs are missing and the `kaggle` CLI isn't "
                "installed. Install it (`pip install kaggle`), set up API credentials "
                "(~/.kaggle/kaggle.json, or KAGGLE_USERNAME/KAGGLE_KEY env vars -- see "
                "https://www.kaggle.com/settings), then retry. Or download "
                "https://www.kaggle.com/datasets/ellipticco/elliptic-data-set by hand and "
                f"unzip it into {root}/"
            )
        os.makedirs(root, exist_ok=True)
        printd('dataset_downloads: fetching elliptic dataset from Kaggle...')
        subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', 'ellipticco/elliptic-data-set',
             '-p', root, '--unzip'],
            check=True,
        )
    # build elliptic_labeled_directed_pyg.pt the way clean_raw.py already does, the same way
    # import_dataset() expects to find it. clean_raw.py assumes cwd is experiments/, so run it
    # with that cwd rather than touching the script itself.
    printd('dataset_downloads: running clean_raw.py to build elliptic_labeled_directed_pyg.pt...')
    experiments_dir = os.path.join(DATASETS_DIR, '..', 'experiments')
    subprocess.run(['python', os.path.join(root, 'clean_raw.py')], cwd=experiments_dir, check=True)


def _no_op():
    '''torch_geometric.datasets.WebKB and pygod.utils.load_data already auto-download on first
    use -- nothing for us to do for these dataset names. Kept explicit so every dataset name
    import_dataset() knows about shows up somewhere in this module.'''
    pass


_DOWNLOADERS = {
    'squirrel': _download_squirrel,
    'cora-ml': lambda: _download_planetoid_npz('cora-ml'),
    'citeseer': lambda: _download_planetoid_npz('citeseer'),
    'JohnsHopkins55': _download_facebook100,
    'photo': lambda: _download_ggad('photo'),
    'reddit': lambda: _download_ggad('reddit'),
    'texas': _no_op,
    'wisconsin': _no_op,
    'cornell': _no_op,
    'disney': _no_op,
    'enron': _no_op,
    'books': _no_op,
    'amazon': _no_op,
}
