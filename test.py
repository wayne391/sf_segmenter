import numpy as np
from sf_segmenter.vis import plot_feats
from sf_segmenter.segmenter import Segmenter

# --- audio --- #
print(' [*] runnung audio...')

import librosa
import soundfile as sf
from sf_segmenter.feature import audio_extract_pcp


# dummy input
# input_feat = np.zeros((442, 12)) # T x F

# real input
path_audio = librosa.util.example_audio_file()
# y, sr = sf.read(path_audio)
y, sr = librosa.load(path_audio)
pcp = audio_extract_pcp(y, sr)

#  init
segmenter = Segmenter()

# run
boundaries = segmenter.process(pcp)

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
    outdir='doc/audio')

print('boundaries:', boundaries)

# --- midi --- # 
print(' [*] runnung midi...')
import miditoolkit
from miditoolkit.midi import parser as mid_parser
from miditoolkit.pianoroll import parser as pr_parser
from sf_segmenter.feature import midi_extract_beat_sync_pianoroll

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
    outdir='doc/midi')

print('boundaries:', boundaries)