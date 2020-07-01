from sf_segmenter.segmenter import Segmenter
import librosa
import miditoolkit

# init
segmenter = Segmenter()

# audio
path_audio = librosa.util.example_audio_file()
boundaries, labs = segmenter.proc_audio(path_audio)
segmenter.plot(outdir='doc/audio')
print('boundaries:', boundaries)
print('labs:', labs)

# midi
# path_midi = miditoolkit.midi.utils.example_midi_file()
path_midi = 'testcases/1430.mid'
boundaries, labs = segmenter.proc_midi(path_midi)
segmenter.plot(outdir='doc/midi')
print('boundaries:', boundaries)
print('labs:', labs)