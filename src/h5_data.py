import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
import multiprocessing as mp
import h5py
import pywt as pw
from scipy import fftpack, signal, stats
import patsy
from statsmodels.robust import mad

import warnings
warnings.filterwarnings("ignore")

TR_PQ = '../data/parquet/train.parquet'
TR_META = '../data/meta/metadata_train.csv'
TS_PQ = '../data/parquet/test.parquet'
TS_META = '../data/meta/metadata_test.csv'

BATCH = 500

WAVELET_TYPE = 'db2'
WAVELET_LEVEL = 14

def denoise_phase(phase):
    wavelet = pw.Wavelet(WAVELET_TYPE)
    wc = pw.wavedec(phase, wavelet, level=WAVELET_LEVEL)
    sigma = mad(wc[-1])
    threshold = sigma * np.sqrt(2 * np.log(len(phase)))

    wc_r = wc[:]
    wc_r[1:] = (pw.threshold(x, threshold) for x in wc[1:])
    return pw.waverec(wc_r, wavelet)

def std_normalize_phase(phase):
    return (phase - np.mean(phase)) / np.std(phase)

def denoise_normalize_phase(phase):
    return std_normalize_phase(denoise_phase(phase))

def get_freq(val, n, d):
    sig_fft = fftpack.fft(val, n=n)
    sample_freq = fftpack.fftfreq(n=n, d=d)
    pos_mask = np.where(sample_freq >= 0)

    freqs = sample_freq[pos_mask][1:]
    power = np.abs(sig_fft)[pos_mask][1:]

    return freqs, power

def get_freq_dom(values, denoised=None, n=1000, d=(0.02 / 800000.)):
    size = n * 2 + 2

    if denoised is None:
        denoised = denoise_phase(values)

    _, p1 = get_freq(values, size, d)
    _, p2 = get_freq(denoised, size, d)

    return np.reshape(np.asarray([p1, p2]).T, (n, 2))

def get_spectrogram(values, denoised=None, fs=1 / (2e-2 / 800000), uselog=True):
    if denoised is None:
        denoised = denoise_phase(values)

    _, _, Sx1 = signal.spectrogram(denoised, fs)
    _, _, Sx2 = signal.spectrogram(values, fs)

    ret = np.concatenate((
        np.reshape(Sx1, (Sx1.shape[0], Sx1.shape[1], -1)),
        np.reshape(Sx2, (Sx2.shape[0], Sx2.shape[1], -1)),
    ),
                         axis=-1)

    if uselog:
        return np.log10(ret)
    else:
        return ret

def metrics(phase, asdict=False):
    f, Pxx = signal.welch(phase)
    ix_mx = np.argmax(Pxx)
    ix_mn = np.argmin(Pxx)

    d = {
        'mean_signal': np.mean(phase),
        'std_signal': np.std(phase),
        'kurtosis_signal': stats.kurtosis(phase),
        'skewness_signal': stats.skew(phase),

        'mean_amp': np.mean(Pxx),
        'std_amp': np.std(Pxx),
        'median_amp': np.median(Pxx),
        'kurtosis_amp': stats.kurtosis(Pxx),
        'skewness_amp': stats.skew(Pxx),

        'max_signal': np.max(phase),
        'min_signal': np.min(phase),

        'max_amp': Pxx[ix_mx],
        'min_amp': Pxx[ix_mn],

        'max_freq': f[ix_mx],
        'min_freq': f[ix_mn],

        'strong_amp': np.sum(Pxx > 2.5),
        'weak_amp': np.sum(Pxx < 0.4),
    }

    if asdict:
        return d
    else:
        return np.asarray(list(d.values()))

def onehot_phase(phase):
    return [0 if phase == 0 else 1, 0 if phase == 1 else 1]

def parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    Like numpy.apply_along_axis(), but takes advantage of multiple
    cores.
    """
    # Effective axis where apply_along_axis() will be applied by each
    # worker (any non-zero axis number would work, so as to allow the use
    # of `np.array_split()`, which is only done on axis 0):
    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)

    # Chunks for the mapping (only a few chunks):
    chunks = [(func1d, effective_axis, sub_arr, args, kwargs)
              for sub_arr in np.array_split(arr, mp.cpu_count())]

    pool = mp.Pool(mp.cpu_count())
    individual_results = pool.map(unpacking_apply_along_axis, chunks)
    # Freeing the workers:
    pool.close()
    pool.join()

    return np.concatenate(individual_results)

def unpacking_apply_along_axis(tp):
    func1d, axis, arr, args, kwargs = tp
    """
    Like numpy.apply_along_axis(), but and with arguments in a tuple
    instead.

    This function is useful with multiprocessing.Pool().map(): (1)
    map() only handles functions that take a single argument, and (2)
    this function can generally be imported from a module, as required
    by map().
    """
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)

def parallel_proc_h5(name, data, meta, batch=1000):
    steps = int(np.ceil(data.shape[0] / batch))

    dir_file = '../data/hdf5/' + name + '.hdf5'
    print('Creating hdf5 file in directory: {}'.format(dir_file))
    hd_file = h5py.File(dir_file, mode='a')

    print('Creating datasets: ', end='')
    denoised_ds = hd_file.create_dataset(
        'denoised',
        shape=data.shape,
        dtype=np.float32,
        chunks=True,
        #compression="gzip",
        #compression_opts=COMPRESSION
	)
    print('denoised, ', end='')

    metrics_ds = hd_file.create_dataset(
        'metrics', shape=(data.shape[0], 19), dtype=np.float16, chunks=True)
    print('metrics, ', end='')

    spectrogram_ds = hd_file.create_dataset(
        'spectrogram',
        shape=(data.shape[0], 129, 3571, 2),
        dtype=np.float32,
        chunks=True,
        #compression="gzip",
        #compression_opts=COMPRESSION
	)
    print('spectrogram, ', end='')

    freq_dom_ds = hd_file.create_dataset(
        'freq_dom',
        shape=(data.shape[0], 1000, 2),
        dtype=np.float32,
        chunks=True)
    print('freq_dom.')

    t = tqdm(range(steps))
    for i in t:
        start = i * batch
        finish = (i + 1) * batch if i + 1 != steps else data.shape[0]

        t.set_description('Denoising')
        denoised_ds[start:finish] = parallel_apply_along_axis(
            denoise_normalize_phase, 1, data[start:finish])

        t.set_description('Metrics')
        ds = parallel_apply_along_axis(metrics, 1, data[start:finish])
        oh = parallel_apply_along_axis(
            onehot_phase, 1, meta['phase'].values[start:finish].reshape(
                finish - start, 1))
        metrics_ds[start:finish] = np.concatenate([ds, oh], axis=1)

        t.set_description('Spectrogram')
        spectrogram_ds[start:finish] = parallel_apply_along_axis(
            get_spectrogram, 1, data[start:finish])

        t.set_description('Frequency')
        freq_dom_ds[start:finish] = parallel_apply_along_axis(
            get_freq_dom, 1, data[start:finish])

    t.close()

if __name__ == '__main__':
    print('Reading train data...')
    data = pq.read_pandas(TR_PQ).to_pandas().values.T
    meta = pd.read_csv(TR_META)
    print('\nTrain processing started...\n')
    parallel_proc_h5('train', data, meta, batch=BATCH)

    print('-'*50)

    print('Reading test data...')
    data = pq.read_pandas(TS_PQ).to_pandas().values.T
    meta = pd.read_csv(TS_META)
    print('\nTest processing started...\n')
    parallel_proc_h5('train', data, meta, batch=BATCH)

    print('Finished')

