import librosa
import soundfile as sf
import numpy as np
from scipy import signal
from scipy.spatial import distance
from scipy.ndimage import filters
# import msaf.utils as U


CONFIG = {
    "M_gaussian": 27,
    "m_embedded": 3,
    "k_nearest": 0.04,
    "Mp_adaptive": 28,
    "offset_thres": 0.05,
    "bound_norm_feats": np.inf  # min_max, log, np.inf,
                                # -np.inf, float >= 0, None
    # For framesync features
    # "M_gaussian"    : 100,
    # "m_embedded"    : 3,
    # "k_nearest"     : 0.06,
    # "Mp_adaptive"   : 100,
    # "offset_thres"  : 0.01
}


def normalize(X, norm_type, floor=0.0, min_db=-80):
    """Normalizes the given matrix of features.
    Parameters
    ----------
    X: np.array
        Each row represents a feature vector.
    norm_type: {"min_max", "log", np.inf, -np.inf, 0, float > 0, None}
        - `"min_max"`: Min/max scaling is performed
        - `"log"`: Logarithmic scaling is performed
        - `np.inf`: Maximum absolute value
        - `-np.inf`: Mininum absolute value
        - `0`: Number of non-zeros
        - float: Corresponding l_p norm.
        - None : No normalization is performed
    Returns
    -------
    norm_X: np.array
        Normalized `X` according the the input parameters.
    """
    if isinstance(norm_type, str):
        if norm_type == "min_max":
            return min_max_normalize(X, floor=floor)
        if norm_type == "log":
            return lognormalize(X, floor=floor, min_db=min_db)
    return librosa.util.normalize(X, norm=norm_type, axis=1)


def median_filter(X, M=8):
    """Median filter along the first axis of the feature matrix X."""
    for i in range(X.shape[1]):
        X[:, i] = filters.median_filter(X[:, i], size=M)
    return X


def gaussian_filter(X, M=8, axis=0):
    """Gaussian filter along the first axis of the feature matrix X."""
    for i in range(X.shape[axis]):
        if axis == 1:
            X[:, i] = filters.gaussian_filter(X[:, i], sigma=M / 2.)
        elif axis == 0:
            X[i, :] = filters.gaussian_filter(X[i, :], sigma=M / 2.)
    return X


def compute_gaussian_krnl(M):
    """Creates a gaussian kernel following Serra's paper."""
    g = signal.gaussian(M, M / 3., sym=True)
    G = np.dot(g.reshape(-1, 1), g.reshape(1, -1))
    G[M // 2:, :M // 2] = -G[M // 2:, :M // 2]
    G[:M // 2, M // 1:] = -G[:M // 2, M // 1:]
    return G


def compute_nc(X):
    """Computes the novelty curve from the structural features."""
    N = X.shape[0]
    # nc = np.sum(np.diff(X, axis=0), axis=1) # Difference between SF's

    nc = np.zeros(N)
    for i in range(N - 1):
        nc[i] = distance.euclidean(X[i, :], X[i + 1, :])

    # Normalize
    nc += np.abs(nc.min())
    nc /= float(nc.max())
    return nc


def pick_peaks(nc, L=16, offset_denom=0.1):
    """Obtain peaks from a novelty curve using an adaptive threshold."""
    offset = nc.mean() * float(offset_denom)
    th = filters.median_filter(nc, size=L) + offset
    #th = filters.gaussian_filter(nc, sigma=L/2., mode="nearest") + offset
    #import pylab as plt
    #plt.plot(nc)
    #plt.plot(th)
    #plt.show()
    # th = np.ones(nc.shape[0]) * nc.mean() - 0.08
    peaks = []
    for i in range(1, nc.shape[0] - 1):
        # is it a peak?
        if nc[i - 1] < nc[i] and nc[i] > nc[i + 1]:
            # is it above the threshold?
            if nc[i] > th[i]:
                peaks.append(i)
    return peaks


def circular_shift(X):
    """Shifts circularly the X squre matrix in order to get a
        time-lag matrix."""
    N = X.shape[0]
    L = np.zeros(X.shape)
    for i in range(N):
        L[i, :] = np.asarray([X[(i + j) % N, j] for j in range(N)])
    return L


def embedded_space(X, m, tau=1):
    """Time-delay embedding with m dimensions and tau delays."""
    N = X.shape[0] - int(np.ceil(m))
    Y = np.zeros((N, int(np.ceil(X.shape[1] * m))))
    for i in range(N):
        # print X[i:i+m,:].flatten().shape, w, X.shape
        # print Y[i,:].shape
        rem = int((m % 1) * X.shape[1])  # Reminder for float m
        Y[i, :] = np.concatenate((X[i:i + int(m), :].flatten(),
                                 X[i + int(m), :rem]))
    return Y


class Segmenter(object):
    def __init__(self, config=CONFIG):
        self.config = config
        
        # collect feats
        self.F = None
        self.E = None
        self.R = None
        self.L = None
        self.SF = None
        self.nc = None
        
    def process(self, F):
        """Main process.
        Returns

        F: feature. T x D
        -------
        est_idxs : np.array(N)
            Estimated times for the segment boundaries in frame indeces.
        est_labels : np.array(N-1)
            Estimated labels for the segments.
        """
        # Structural Features params
        Mp = self.config["Mp_adaptive"]   # Size of the adaptive threshold for
                                          # peak picking
        od = self.config["offset_thres"]  # Offset coefficient for adaptive
                                          # thresholding
        M = self.config["M_gaussian"]     # Size of gaussian kernel in beats
        m = self.config["m_embedded"]     # Number of embedded dimensions
        k = self.config["k_nearest"]      # k*N-nearest neighbors for the
                                          # recurrence plot

        # Normalize
        F = normalize(F, norm_type=self.config["bound_norm_feats"])

        # Check size in case the track is too short
        if F.shape[0] > 20:
            # Emedding the feature space (i.e. shingle)
            E = embedded_space(F, m)
           
            # Recurrence matrix
            R = librosa.segment.recurrence_matrix(
                E.T,
                k=k * int(F.shape[0]),
                width=1,  # zeros from the diagonal
                metric="euclidean",
                sym=True).astype(np.float32)

            # Circular shift
            L = circular_shift(R)

            # Obtain structural features by filtering the lag matrix
            SF = gaussian_filter(L.T, M=M, axis=1)
            SF = gaussian_filter(SF, M=1, axis=0)

            # Compute the novelty curve
            nc = compute_nc(SF)

            # Find peaks in the novelty curve
            est_bounds = pick_peaks(nc, L=Mp, offset_denom=od)

            # Re-align embedded space
            est_bounds = np.asarray(est_bounds) + int(np.ceil(m / 2.))
        else:
            est_bounds = []

        # Add first and last frames
        est_idxs = np.concatenate(([0], est_bounds, [F.shape[0] - 1]))
        est_idxs = np.unique(est_idxs)

        assert est_idxs[0] == 0 and est_idxs[-1] == F.shape[0] - 1
        
        # collect  feature
        self.F = F
        self.E = E
        self.R = R
        self.L = L
        self.SF = SF
        self.nc = nc
        
        return est_idxs


if __name__ == '__main__':
    from vis import plot_feats
    # --- audio --- #
    # import soundfile as sf
    # from feature import audio_extract_pcp
    # # dummy input
    # input_feat = np.zeros(442, 12) # T x F
    
    # real input
    # path_audio = librosa.util.example_audio_file()
    # y, sr = sf.read(path_audio)
    # pcp = audio_extract_pcp(y, sr)

    # #  init
    # segmenter = Segmenter()

    # # run
    # boundarues = segmenter.process(pcp)

    # --- midi --- # 
    import miditoolkit
    from miditoolkit.midi import parser as mid_parser
    from miditoolkit.pianoroll import parser as pr_parser
    from feature import midi_extract_beat_sync_pianoroll

    # parse midi to pianoroll
    path_midi = miditoolkit.midi.utils.example_midi_file()
    midi_obj = mid_parser.MidiFile(path_midi)

    notes = midi_obj.instruments[0].notes
    pianoroll = pr_parser.notes2pianoroll(
                        notes)

    # pianoroll to beat sync pianoroll
    pianoroll_sync = midi_extract_beat_sync_pianoroll(
            pianoroll,
            midi_obj.ticks_per_beat)  

    print('pianoroll_sync :', pianoroll_sync.shape)
    #  init
    segmenter = Segmenter()

    # run
    boundaries = segmenter.process(pianoroll_sync)

    # vis
    data = {
        'input': segmenter.F,
        'R': segmenter.R,
        'L': segmenter.L,
        'SF': segmenter.SF,
        'nc': segmenter.nc,
    }
    plot_feats(data, 
        boundaries, 
        outdir='doc')

    print('boundaries:', boundaries)