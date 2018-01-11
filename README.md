
# Wavelet prosody analyzer

antti.suni@helsinki.fi

## Description

The program calculates f0, energy and duration features from speech wav-file, performs continuous
wavelet analysis on combined features, finds prosodic events (prominences, boundaries) from
the wavelet scalogram and aligns the events with transcribed units.

See also:
 "Hierarchical representation and estimation of prosody using continuous wavelet transform"
 http://www.sciencedirect.com/science/article/pii/S0885230816303527

The default settings of the program are roughly the same as in the paper,
duration signal was generated from word level labels.



## Requirements

see INSTALL.txt
input: audio files in wav format, transcriptions in either htk .lab format or Praat textgrids


## Usage:
1. to start, type
```sh
python wavelet_gui.py
```

2. Select directory with speech and transciption files: 'Select Speech Directory...'
Some examples are provided in samples/ directory. Files should have the same root,
for example file1.wav, file1.lab  or file2.wav file2.TextGrid.

3. Select features to use in analysis: 'Prosodic Feats for CWT..'

4. Adjust Pitch tracking parameters for the speaker / environment, press 'Reprocess' to see changes
Set range for possible pitch values, typically males ~50-350Hz, females ~100-400Hz.
If estimated track skips obviously voiced portions, move voicing threshold slider left.

-Alternatively, pre-estimated f0 analyses can be used: file <file>.f0 must exist and it should be
either in praat matrix format or as a list file with one f0 value / line, frame shift must be constant 5ms.
(To get suitable format from Praat, select wav <myfile> and do:
    To Pitch: 0.005, 120, 400
    To Matrix
    Save as matrix text file: "<mydirectory>/<myfile>.f0"
)


5. Adjust the weights of prosodic features and choose if the final signal is combined by summing or multiplying the features

6. Select which tiers to use for durations signal generation / use duration estimated from signal

7. Select transcription level of interest: 'Select Tier'

7. You can interactively zoom and move around with the button on top, and play the visible section

8. When everything is good, you can 'Process all' which analyzes all utterances in the directory
with the current settings, and saves prosodic labels  as <file_name_root>.prom
Prosodic labels are saved in a tab separated form with the following columns:
<unit_start> <unit_end> <unit> <prominence strength> <prominence_position> <boundary strength>
