import librosa
import soundfile as sf
import numpy as np
from scipy import signal
from scipy.spatial import distance
from scipy.ndimage import filters

from .vis import plot_feats
from .feature import midi_extract_beat_sync_pianoroll, audio_extract_pcp


import miditoolkit
from miditoolkit.midi import parser as mid_parser
from miditoolkit.pianoroll import parser as pr_parser

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


def cummulative_sum_Q(R):
    len_x, len_y = R.shape
    Q = np.zeros((len_x + 2, len_y + 2))
    for i in range(len_x):
        for j in range(len_y):
            Q[i+2, j+2] = max(
                    Q[i+1, j+1],
                    Q[i, j+1],
                    Q[i+1, j]) + R[i, j]
    return np.max(Q)
    

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


def run_label(
        boundaries, 
        R,
        max_iter=100, 
        return_feat=False):

    """Labeling algorithm."""
    n_boundaries = len(boundaries)
    
    # compute S
    S = np.zeros((n_boundaries, n_boundaries))
    for i in range(n_boundaries - 1):
        for j in range(n_boundaries - 1):
            i_st, i_ed = boundaries[i], boundaries[i+1]
            j_st, j_ed = boundaries[j], boundaries[j+1]

            len_i = i_ed - i_st
            len_j = j_ed - j_st
            score = cummulative_sum_Q(R[i_st:i_ed, j_st:j_ed])
            S[i, j] = score / min(len_i, len_j)    
    
    # threshold
    thr = np.std(S) + np.mean(S)
    S[S <= thr] = 0       
    
    # iteration
    S_trans = S.copy()
    
    for i in range(max_iter):
        S_trans = np.matmul(S_trans, S)
        
    S_final = S_trans > 1
    
    # proc output
    n_seg = len(S_trans) - 1
    labs = np.ones(n_seg) * -1
    cur_tag = int(-1)
    for i in range(n_seg):
        print(' >', i)
        if labs[i] == -1:
            cur_tag += 1
            labs[i] = cur_tag
            for j in range(n_seg):
                if S_final[i, j]:
                    labs[j] = cur_tag
    
    if return_feat:
        return labs, (S, S_trans, S_final)
    else:
        return labs


class Segmenter(object):
    def __init__(self, config=CONFIG):
        self.config = config
        self.refresh()

    def refresh(self):
        # collect feats
        # - segmentation
        self.F = None
        self.E = None
        self.R = None
        self.L = None
        self.SF = None
        self.nc = None

        # - labeling
        self.S = None
        self.S_trans = None
        self.S_final = None

        # - res
        self.boundaries = None
        self.labs = None


    def proc_midi(self, path_midi, is_label=True):
        # parse midi to pianoroll
        midi_obj = mid_parser.MidiFile(path_midi)
        notes = midi_obj.instruments[0].notes
        pianoroll = pr_parser.notes2pianoroll(
                            notes)

        # pianoroll to beat sync pianoroll
        pianoroll_sync = midi_extract_beat_sync_pianoroll(
                pianoroll,
                midi_obj.ticks_per_beat)  

        return self.process(pianoroll_sync, is_label=is_label)

    def proc_audio(self, path_audio, sr=22050, is_label=True):
        y, sr = librosa.load(path_audio, sr=sr)
        pcp = audio_extract_pcp(y, sr)
        return self.process(pcp, is_label=is_label)

        
    def process(
            self, 
            F, 
            is_label=False):
        """Main process.
        Returns

        F: feature. T x D
        """
        self.refresh()

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

        if is_label:
            labs, (S, S_trans, S_final) = run_label(
                est_idxs, 
                R,
                return_feat=True)

            self.S = S
            self.S_trans = S_trans
            self.S_final = S_final
            
            self.boundaries = est_idxs
            self.labs = labs
            return est_idxs, labs
        else:
            self.boundaries = est_idxs
            return est_idxs

    def plot(
            self, 
            outdir=None,
            vis_bounds=True):

        plot_feats(
            F=self.F,
            R=self.R,
            L=self.L,
            SF=self.SF,
            nc=self.nc,
            S=self.S,
            S_trans=self.S_trans,
            S_final=self.S_final,
            boundaries=self.boundaries,
            outdir=outdir,
            vis_bounds=vis_bounds)
