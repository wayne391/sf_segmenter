import librosa
import numpy as np
from miditoolkit.pianoroll import utils as mt_utils


def audio_extract_pcp(
        audio, 
        sr,
        n_fft=4096,
        hop_len=int(4096 * 0.75),
        pcp_bins=84,
        pcp_norm=np.inf,
        pcp_f_min=27.5,
        pcp_n_octaves=6):

    audio_harmonic, _ = librosa.effects.hpss(audio)
    pcp_cqt = np.abs(librosa.hybrid_cqt(
                audio_harmonic,
                sr=sr,
                hop_length=hop_len,
                n_bins=pcp_bins,
                norm=pcp_norm,
                fmin=pcp_f_min)) ** 2

    pcp = librosa.feature.chroma_cqt(
                C=pcp_cqt,
                sr=sr,
                hop_length=hop_len,
                n_octaves=pcp_n_octaves,
                fmin=pcp_f_min).T
    return pcp


def midi_extract_beat_sync_pianoroll(
        pianoroll,
        beat_resol,
        is_tochroma=False):

    # sync to beat
    beat_sync_pr = np.zeros(
        (int(np.ceil(pianoroll.shape[0] /  beat_resol)),
         pianoroll.shape[1]))

    for beat in range(beat_sync_pr.shape[0]):
        st = beat * beat_resol
        ed = (beat + 1) * beat_resol
        beat_sync_pr[beat] = np.sum(pianoroll[st:ed, :], axis=0)
    
    # normalize
    beat_sync_pr = (
        beat_sync_pr - beat_sync_pr.mean()) / beat_sync_pr.std()
    beat_sync_pr = (
        beat_sync_pr - beat_sync_pr.min()) / (beat_sync_pr.max() - beat_sync_pr.min())

    # to chroma
    if is_tochroma:
        beat_sync_pr = mt_utils.tochroma(beat_sync_pr)
    return beat_sync_pr
