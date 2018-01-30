#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUTHOR
    - Antti Suni <antti.suni@helsinki.fi>

DESCRIPTION

LICENSE
    This script is in the public domain, free from copyrights or restrictions.
    Created: 21 January 2018
"""

import sys
import os
import traceback
import argparse
import time
import logging

# QT related imports
from PyQt5 import QtGui,QtCore, QtWidgets, QtMultimedia
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Plotting configuration
from matplotlib.ticker import MaxNLocator
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

# Numpy
import numpy as np

# Types
import types

# CSV helpers
import csv

# Wavelet part
# - acoustic features
from wavelet_prosody_toolkit.prosody_tools import f0_processing, energy_processing, duration_processing

# - helpers
from wavelet_prosody_toolkit.prosody_tools import misc, smooth_and_interp, pitch_tracker

# - wavelet transform
from wavelet_prosody_toolkit.prosody_tools import cwt_utils, loma, lab

# Globbing
import os,glob

# Python 3 compatibility hack
try:
    unicode('')
except NameError:
    unicode = str


# Analysis sample rate
ANALYSIS_SR = 8000.0

# Plot sample rate
PLOT_SR = 200.0

# List of logging levels used to setup everything using verbose option
LEVEL = [logging.WARNING, logging.INFO, logging.DEBUG]

##############################################################################################
## Callbacks
##############################################################################################
def press_zoom(self, event):
    """Zoom call back

    Constraint the zoom to the x-axis only

    Parameters
    ----------
    self: type
        description
    event: type
        description

    """
    event.key='x'
    NavigationToolbar.press_zoom(self,event)

def drag_pan(self, event):
    """Drag pan callback

    Constraint the pan to the x-axis only

    Parameters
    ----------
    self: type
        description
    event: type
        description
    """
    event.key='x'
    NavigationToolbar.drag_pan(self,event)



##############################################################################################
## Window class
##############################################################################################


class SigWindow(QtWidgets.QDialog):
    """Main window class
    """

    def __init__(self, parent=None):
        """Initialisation method for the new created window

        Parameters
        ----------
        self: SigWindow
            The freshly created window object
        parent: Parent window
            The parent object of the current window [default=None]

        """
        super(SigWindow, self).__init__(parent)

        # Define the logger
        self.logger = logging.getLogger(__name__)


        ##########################################
        # Define internal variables
        ##########################################
        self.dir = '.'
        self.wav_files = []
        self.cur_wav = None

        self.energy = []
        self.F0 = []
        self.duration = []
        self.params = []
        self.fUpdate = {}
        for f in ['wav', 'energy', 'f0', 'duration', 'params', 'tiers', 'cwt', 'loma']:
            self.fUpdate[f] = True
        self.fProcessAll=False
        self.fUsePrecalcF0 = True
        self.current_tier_index = -1
        self.current_dur_tier_indices = []


        ##########################################
        # Setup the plot area
        ##########################################
        plt.rcParams['xtick.major.pad'] = 8
        plt.rcParams['ytick.major.pad'] = 8

        self.figure = plt.figure()

        self.ax = []
        self.ax.append(plt.subplot(611))
        self.ax.append(plt.subplot(612,sharex=self.ax[0]))
        self.ax.append(plt.subplot(613,sharex=self.ax[0]))
        self.ax.append(plt.subplot(6,1,(4,6),sharex=self.ax[0]))

        self.ax[0].set_ylabel("Spec")
        self.ax[1].set_ylabel("F0")
        self.ax[2].set_ylabel("Signals")
        self.ax[3].set_ylabel("Wavelet")

        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(400, 400)


        ##########################################
        # Setup the toolbar
        ##########################################
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.press_zoom=types.MethodType(press_zoom, self.toolbar)
        self.toolbar.drag_pan=types.MethodType(drag_pan, self.toolbar)


        ##########################################
        # Setup the status bar
        ##########################################
        self.status = QtWidgets.QStatusBar()
        self.status.setMaximumSize(800,30)
        self.status.showMessage("Wavelet Prosody Analyzer | to start, find a folder with audio files and associated labels ")

        ##########################################
        # Define the left part of the window
        ##########################################
        left_layout = QtWidgets.QVBoxLayout()
        left_layout.addWidget(self.toolbar)
        left_layout.addWidget(self.canvas)
        left_layout.addWidget(self.status)

        ##########################################
        # Define directory listing
        ##########################################
        self.filelist = QtWidgets.QListWidget(self)
        self.filelist.setMaximumSize(800,300)
        self.filelist.currentItemChanged.connect(self.onWavChanged)


        # Define directory selection button
        self.chooseDir =QtWidgets.QPushButton('Select Speech Directory')
        self.chooseDir.clicked.connect(self.dirDialog)
        self.chooseDir.setDefault(False)
        self.chooseDir.setAutoDefault(False)

        # Define process button
        self.bProcessAll = QtWidgets.QPushButton("Process all files", self)
        self.bProcessAll.clicked.connect(self.processAll)
        self.bProcessAll.setToolTip("Annotate all speech files in the selected folder with current settings")
        self.bProcessAll.setDefault(False)
        self.bProcessAll.setAutoDefault(False)

        # Define reprocess button
        self.reprocess = QtWidgets.QPushButton('Reprocess')
        self.reprocess.clicked.connect(self.onReprocess)
        self.reprocess.setDefault(False)
        self.reprocess.setAutoDefault(False)


        # Define play button
        self.bPlay = QtWidgets.QPushButton("Play", self)
        self.bPlay.clicked.connect(self.play)
        self.bPlay.setDefault(False)
        self.bPlay.setAutoDefault(False)

        # Define existing F0 checkbox
        self.bUseExistingF0 = QtWidgets.QCheckBox("Use existing F0 files if available")
        self.bUseExistingF0.clicked.connect(self.onF0Changed)
        self.bUseExistingF0.setToolTip("See examples folder for supported formats")


        ##########################################
        # Define the right part of the window
        ##########################################
        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(self.filelist)
        right_layout.addWidget(self.chooseDir)
        right_layout.addWidget(self.bProcessAll)
        right_layout.addWidget(self.setF0Limits())
        right_layout.addWidget(self.bUseExistingF0)
        right_layout.addWidget(self.prosodicFeats())
        right_layout.addWidget(self.reprocess)
        right_layout.addWidget(self.featureCombination())
        right_layout.addWidget(self.weight())
        right_layout.addWidget(self.signalTiers())
        right_layout.addWidget(self.createTierList())
        right_layout.addWidget(self.bPlay)


        ##########################################
        # Finalize the window layout
        ##########################################
        layout = QtWidgets.QHBoxLayout()
        layout.addLayout(left_layout,3)
        layout.addLayout(right_layout,1)
        self.setLayout(layout)


    def setF0Limits(self):
        """Setup the F0 limits area

        Parameters
        ----------
        self: SigWindow
            The current window object

        Returns
        -------
        groupBox: QGroupBox
            The groupbox containing all the controls needed for F0 limit definition
        """

        # Min F0
        self.min_f0 = QtWidgets.QLineEdit("min F0")
        self.min_f0.setText("50")
        self.min_f0.setInputMask("000")
        self.min_f0.textChanged.connect(self.onF0Changed)

        # Max F0
        self.max_f0 = QtWidgets.QLineEdit("min F0")
        self.max_f0.setText("400")
        self.max_f0.setInputMask("000")
        self.max_f0.textChanged.connect(self.onF0Changed)

        # Voicing
        self.voicing = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.voicing.setSliderPosition(50)
        self.voicing.valueChanged.connect(self.onF0Changed)

        # Harmonics
        self.harmonics = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.harmonics.setSliderPosition(50)
        self.harmonics.setVisible(False)
        #self.harmonics.valueChanged.connect(self.onF0Changed)

        # Setup groupbox
        hbox = QtWidgets.QVBoxLayout()
        hbox.addWidget(self.min_f0)
        hbox.addWidget(self.max_f0)
        hbox.addWidget(self.voicing)
        #hbox.addWidget(self.harmonics)

        groupBox = QtWidgets.QGroupBox("minF0, maxF0, voicing threshold") #, harmonics")
        #groupBox.setMaximumSize(200,200)
        groupBox.setLayout(hbox)
        groupBox.setToolTip("min and max Hz of the speaker's f0 range, voicing threshold")

        return groupBox

    def prosodicFeats(self):
        """Function to setup the feature weights for the CWT

        Parameters
        ----------
        self: SigWindowtype
            The current window object

        Returns
        -------
        groupBox: QGroupBox
            The groupbox containing all the controls related to the feature weights

        """
        groupBox = QtWidgets.QGroupBox("Feature Weights for CWT")

        # F0 widgets
        l1 = QtWidgets.QLabel("F0")
        self.wF0 = QtWidgets.QLineEdit("1.0")
        self.wF0.setInputMask("0.0")
        self.wF0.setMaxLength(3)

        # Energy widgets
        l2 = QtWidgets.QLabel("Energy")
        self.wEnergy = QtWidgets.QLineEdit("1.0")
        self.wEnergy.setInputMask("0.0")
        self.wEnergy.setMaxLength(3)

        # Duration widgets
        l3 = QtWidgets.QLabel("Duration")
        self.wDuration = QtWidgets.QLineEdit("1.0")
        self.wDuration.setInputMask("0.0")
        self.wDuration.setMaxLength(3)


        # Setup the groupbox
        box = QtWidgets.QGridLayout()
        box.addWidget(l1, 0,0)
        box.addWidget(l2, 0,1)
        box.addWidget(l3, 0,2)
        box.addWidget(self.wF0, 1,0)
        box.addWidget(self.wEnergy, 1,1)
        box.addWidget(self.wDuration, 1,2)
        groupBox.setLayout(box)

        return groupBox

    def signalTiers(self):
        """?

        Parameters
        ----------
        self: SigWindowtype
            The current window object

        Returns
        -------
        groupBox: QGroupBox
            The groupbox containing all the controls related to the feature weights

        """

        # Signal tier
        self.signalTiers = QtWidgets.QListWidget()
        self.signalTiers.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.signalTiers.clicked.connect(self.onSignalRate)

        # Signal rate
        self.signalRate = QtWidgets.QCheckBox("Estimate speech rate from signal")
        self.signalRate.setChecked(False)
        self.signalRate.clicked.connect(self.onSignalRate)

        # Delta
        self.diffDur = QtWidgets.QCheckBox("Use delta-duration")
        self.diffDur.setToolTip("Point-wise difference of the durations signal, empirically found to improve boundary detection in some cases")
        self.diffDur.clicked.connect(self.onSignalRate)

        # Setup the group box
        box = QtWidgets.QVBoxLayout()
        box.addWidget(self.signalTiers)
        box.addWidget(self.diffDur)
        box.addWidget(self.signalRate)
        groupBox = QtWidgets.QGroupBox("Tier(s) for Duration Signal")
        groupBox.setMaximumSize(400,150) # FIXME: see for not having hardcoded size
        groupBox.setLayout(box)
        groupBox.setToolTip("Generate duration signal from a tier or as a sum of two or more tiers.\nShift-click to multi-select, Ctrl-click to de-select")

        return groupBox

    def weight(self):
        groupBox=QtWidgets.QGroupBox("frequency / time resolution")
        groupBox.setToolTip("Interpolation between Mexican Hat wavelet (left) and Gaussian filter / scale-space (right).")
        self.weight = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.weight.sliderReleased.connect(self.onWeightChanged)

        hbox = QtWidgets.QVBoxLayout()
        hbox.addWidget(self.weight)
        groupBox.setLayout(hbox)
        groupBox.setVisible(False)
        return groupBox

    def featureCombination(self):


        groupBox =  QtWidgets.QGroupBox("Feature Combination Method")

        combination_method =QtWidgets.QButtonGroup() # Number group

        self.sum_feats=QtWidgets.QRadioButton("sum")
        self.mul_feats=QtWidgets.QRadioButton("product")
        self.sum_feats.setChecked(True)
        combination_method.addButton(self.sum_feats)
        combination_method.addButton(self.mul_feats)
        self.sum_feats.clicked.connect(self.onSignalRate)
        self.mul_feats.clicked.connect(self.onSignalRate)
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.sum_feats)
        hbox.addWidget(self.mul_feats)
        groupBox.setLayout(hbox)
        groupBox.setVisible(True)
        return groupBox


    # reading of textgrids and labs
    def populateTierList(self):
        import os.path

        # remember current tier selections


        current_tier = self.tierlist.currentIndex()
        current_dur_tiers = [x.row() for x in self.signalTiers.selectedIndexes()]
        if current_tier >=0:
            self.current_tier_index = current_tier
        if len(current_dur_tiers) > 0:
            self.current_dur_tier_indices = current_dur_tiers


        current = unicode(self.tierlist.currentText())
        current_index = self.tierlist.currentIndex()
        current2 = [item.text() for item in self.signalTiers.selectedItems()]

        # clear selection
        self.tierlist.clear()
        self.signalTiers.clear()
        self.tiers = {}

        self.logger.debug("reading labels..")
        # read htk lab or textgrid
        lab_f = os.path.splitext(unicode(self.cur_wav))[0]+".lab"
        if os.path.exists(lab_f):
            try:
                self.tiers = lab.read_htk_label(lab_f)
            except:
                pass
        if not self.tiers:
            grid = os.path.splitext(unicode(self.cur_wav))[0]+".TextGrid"
            if os.path.exists(grid):
                self.tiers = lab.read_textgrid(grid)
            else:
                logging.error(grid +" not found")
        if not self.tiers:
            return


        for k in sorted(self.tiers.keys()):
            self.tierlist.addItem(k)
            self.signalTiers.addItem(k)
        try:

            if self.current_tier_index >= 0:
                self.tierlist.setCurrentIndex(self.current_tier_index)
        except:
            try:
                self.signalTiers.setCurrentIndex(0)
                self.tierlist.setCurrentIndex(0)
            except:
                pass
        if len(self.current_dur_tier_indices) > 0:

            for i in self.current_dur_tier_indices:
                try:
                    cur=self.signalTiers.item(i)
                    cur.setSelected(True)
                except:
                    pass
        else:
            self.signalTiers.item(0).setSelected(True)



    def createTierList(self):
        groupBox = QtWidgets.QGroupBox("Tier for Prosody Annotation")
        self.tierlist = QtWidgets.QComboBox()
        self.tierlist.activated.connect(self.onTierChanged)
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.tierlist)
        groupBox.setLayout(vbox)
        return groupBox


    ##############################################################
    ## Event callbacks
    ##############################################################
    def onWeightChanged(self):
        self.fUpdate['cwt']=True
        self.analysis()

    def onTierChanged(self, i):
        self.fUpdate['tiers']=True
        #self.fUpdate['loma']=True
        self.analysis()


    def onF0Changed(self):
        self.fUpdate['f0']=True
        self.analysis()

    def onSignalRate(self):
        if self.signalRate.isChecked():
            self.signalTiers.setEnabled(False)
        else:
            self.signalTiers.setEnabled(True)
        self.fUpdate['duration']=True
        self.analysis()

    def onWavChanged(self, curr, prev):
        if not curr:
            return
        self.cur_wav = self.dir+'/'+unicode(curr.text())
        self.status.showMessage("Wavelet Prosody Analyzer | processing " +curr.text()+"...")
        self.populateTierList()
        time.sleep(0.05)
        QtWidgets.qApp.processEvents()
        self.fUpdate = dict.fromkeys(self.fUpdate, True)
        self.analysis()
        self.status.showMessage("Wavelet Prosody Analyzer | "+curr.text())

    def onReprocess(self):
        self.fUpdate['params']=True
        self.analysis()

    def refresh_updates(self):
        for f in ['duration', 'f0', 'energy', 'wav']:
            if self.fUpdate[f]:
                self.fUpdate['params']=True

        if self.fUpdate['params']:
            self.fUpdate['cwt']=True
            self.fUpdate['loma']=True

        if self.fUpdate['tiers']:
            self.fUpdate['loma']=True

    #
    def processAll(self):
        """batch processing of whole directory
        """
        results = []
        if not self.fProcessAll:
            self.fProcessAll=True
            self.bProcessAll.setText("Stop Processing")
        else:
            self.fProcessAll=False
            self.bProcessAll.setText("Process All Files")
            return

        for i in range(self.filelist.count()):

            if not self.fProcessAll:
                break

            # this triggers the analysis
            self.filelist.setCurrentRow(i)


            # get results
            feats = [unicode(self.filelist.currentItem().text())]
            prominences= np.array_str(self.prominences[:,1], precision=3) #sprintf("%0.3f" % self.prominences)
            for p in self.prominences:
                feats.append("%0.5f" %p[1])

            # if one line per utterance:
            res_str = unicode(self.filelist.currentItem().text())+u"\t".join(self.prominences[:,1].astype('unicode')) #.join("\t")
            results.append(feats) #self.prominences)

            # if one file per utterance:

            prom_f = os.path.splitext(unicode(self.cur_wav))[0]+".prom"
            self.logger.debug(feats)
            time.sleep(0.05)





        self.logger.debug("writing results to "+ self.dir+"/results.txt")
        with open(self.dir+"/results.txt", 'w') as res_file:
            writer=csv.writer(res_file, delimiter='\t')
            writer.writerows(results)

        self.logger.debug("written")
        self.status.showMessage("Wavelet Prosody Analyser | analyses saved in "+self.dir+"/results.txt")
        self.fProcessAll=False
        self.bProcessAll.setText("Process All Files")


    def dirDialog(self):

        dirname = str(QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Directory', self.dir)) #os.getcwd()))
        self.wav_files = glob.glob(dirname+'/*.wav') #[Wv][Aa][Wv]') #(WAV)|(wav)')
        self.dir = dirname
        self.filelist.clear()
        for i in range(len(self.wav_files)):
            #self.filelist.addItem(os.path.basename(self.wav_files[i].decode('utf-8'))) #'Item %s' % (i + 1))
            self.filelist.addItem(os.path.basename(self.wav_files[i])) #'Item %s' % (i + 1))
        if len(self.wav_files) > 0:
            self.status.showMessage("processing " + self.wav_files[i])
            QtWidgets.qApp.processEvents()
            self.filelist.setCurrentRow(0)




    def play(self):
        # todo: find python method for this,
        # sox usage for windows probably difficult . done

        import tempfile

        # get the current selection
        (st, end) =plt.gca().get_xlim()
        st=np.max([0,st])
        self.logger.debug(st, end)
        st/=PLOT_SR
        end/=PLOT_SR

        # save to tempfile
        #bug: cuts from the end?
        wav_slice = self.sig[int(st*self.orig_sr):int(end*self.orig_sr)]
        fname = tempfile.mkstemp()[1]
        misc.write_wav(fname, wav_slice, self.orig_sr)


        # NOTE: QSound.play used to fail silently on some systems

        try:
            QtMultimedia.QSound.play(fname) #unicode(self.cur_wav))
        except:
            self.logger.debug("Qsound does not play")
            os.system("play "+fname)

    def get_float_val(self, qt_obj):
        return float(qt_obj.text())

    # main function
    # analysis and plotting of acoustic features and wavelets + loma
    def analysis(self):
        prev_zoom = None

        if not self.fUpdate["wav"]:
            prev_zoom = self.ax[3].axis()

        if not self.cur_wav:
            return

        self.refresh_updates()
        # show spectrogram
        if self.fUpdate['wav']:
            self.toolbar.update()
            self.logger.debug("plot specgram")

            self.ax[0].cla()
            self.orig_sr, self.sig = misc.read_wav(self.cur_wav)
            #downsample = int(self.orig_sr /ANALYSIS_SR)
            #self.sig = misc.resample(self.sig, len(self.sig)/downsample)
            #self.orig_sr = ANALYSIS_SR
            self.plot_len = int(len(self.sig)*(PLOT_SR/self.orig_sr))
            self.ax[0].specgram(self.sig,NFFT=200,noverlap=40, Fs = self.orig_sr,xextent=[0, self.plot_len], cmap="jet")


        if self.fUpdate['energy']:
            # 'energy' is just a smoothed envelope here
            self.logger.debug("analyzing energy..")
            self.energy = energy_processing.extract_energy(self.sig, self.orig_sr, 300, 5000)
            #self.energy_smooth = smooth_and_interp.peak_smooth(self.energy, 30, 3)
            self.energy_smooth = smooth_and_interp.peak_smooth(self.energy, 30, 3)
            #self.energy_smooth = self.energy
        raw_pitch = None
        if self.fUpdate['f0']:
            self.ax[1].cla()
            self.pitch = None
            raw_pitch = None
            # if f0 file is provided, use that
            if self.bUseExistingF0.isChecked():
                raw_pitch = f0_processing.read_f0(self.cur_wav)

            # else use reaper
            if raw_pitch is None:
                # analyze pitch
                self.logger.debug("analyzing pitch..")
                min_f0 = float(str(self.min_f0.text()))
                max_f0 = float(str(self.max_f0.text()))

                #raw_pitch2 = f0_processing.extract_f0(self.cur_wav, self.sig, self.orig_sr, min_f0, max_f0)
                (raw_pitch, pic) = pitch_tracker.inst_freq_pitch(self.cur_wav,min_f0, max_f0, float(self.harmonics.value()),float(self.voicing.value()))
                # fix errors, smooth and interpolate
            try:
                self.pitch = f0_processing.process(raw_pitch)
            except:
                # f0_processing.process crashes if raw_pitch is all zeros, kludge
                self.pitch = raw_pitch
            #self.pitch2 = f0_processing.process(raw_pitch2)
            #self.ax[1].plot(raw_pitch2,color='red', linewidth=1)
            #self.ax[1].plot(self.pitch2,color='red', linewidth=2)
            self.ax[1].plot(raw_pitch,color='black', linewidth=1)
            self.ax[1].plot(self.pitch,color='black', linewidth=2)
            self.ax[1].set_ylim(np.min(self.pitch)*0.75, np.max(self.pitch)*1.2)



        if self.fUpdate['duration']:
            self.logger.debug("analyzing duration...")

            # signal method for speech rate, quite shaky
            if self.signalRate.isChecked():
                self.rate = duration_processing.get_rate(self.energy) #, fig=self.ax[2])
                self.rate = smooth_and_interp.smooth(self.rate, 30)

            # word / syllable / segment duration from labels
            else:
                sig_tiers = []
                for item in self.signalTiers.selectedItems():
                    sig_tiers.append(self.tiers[item.text()])

                try:
                    self.rate = duration_processing.get_duration_signal(sig_tiers)
                except:
                    self.rate = np.zeros(len(self.pitch))

            if self.diffDur.isChecked():
                self.rate = np.diff(self.rate,1)

            try:
                self.rate = np.pad(self.rate, (0,len(self.pitch)-len(self.rate)), 'edge')
            except:
                self.rate = self.rate[0:len(self.pitch)]


        # combine acoustic features by normalizing, fixing lengths and summing (or multiplying)
        if self.fUpdate['params'] ==True:
            self.ax[2].cla()
            self.ax[3].cla()
            self.ax[2].plot(misc.normalize_std(self.pitch)+12, label="F0")
            self.ax[2].plot(misc.normalize_std(self.energy_smooth)+8, label="Energy")
            #self.ax[2].plot(misc.normalize_std(self.energy)+8, label="Energy")
            self.ax[2].plot(misc.normalize_std(self.rate)+4, label="Duration")


            self.energy_smooth = self.energy_smooth[:np.min([len(self.pitch), len(self.energy_smooth)])]
            self.pitch = self.pitch[:np.min([len(self.pitch), len(self.energy_smooth)])]
            self.rate = self.rate[:np.min([len(self.pitch), len(self.rate)])]


            if self.mul_feats.isChecked():

                pitch = np.ones(len(self.pitch))
                energy = np.ones(len(self.pitch))
                duration =  np.ones(len(self.pitch))
                if self.get_float_val(self.wF0) > 0:

                    pitch = misc.normalize_minmax(self.pitch)+self.get_float_val(self.wF0)
                if self.get_float_val(self.wEnergy)> 0:
                    energy = misc.normalize_minmax(self.energy_smooth)+self.get_float_val(self.wEnergy)
                if self.get_float_val(self.wDuration)>0:
                    duration = misc.normalize_minmax(self.rate)+self.get_float_val(self.wDuration)

                params = pitch * energy * duration


            else:
                params = misc.normalize_std(self.pitch)*float(self.wF0.text()) + \
                         misc.normalize_std(self.energy_smooth)*float(self.wEnergy.text()) + \
                         misc.normalize_std(self.rate)*float(self.wDuration.text())
            #params = smooth_and_interp.remove_bias(params, 800)
            self.params = misc.normalize_std(params)
            self.ax[2].plot(params,color="black", linewidth=2, label="Combined")


        try:
            labels = self.tiers[unicode(self.tierlist.currentText())]
        except:
            labels = None

        if self.fUpdate['tiers']:
            self.ax[3].cla()

        # do wavelet analysis
        #n_scales = 22
        n_scales = 40
        scale_dist = 0.25
        if self.fUpdate['cwt']:
            self.logger.debug("wavelet transform...")

            self.fEnergy = False
            if not self.fEnergy:
                (self.cwt,self.scales) = cwt_utils.cwt_analysis(self.params, mother_name="mexican_hat",period=2,num_scales=n_scales, scale_distance=scale_dist,apply_coi=True)
                self.cwt = np.real(self.cwt)
            else:
                (self.cwt,self.scales) = cwt_utils.cwt_analysis(params, mother_name="morlet",period=5,num_scales=n_scales, scale_distance=scale_dist,apply_coi=False)
                self.cwt = np.abs(self.cwt)
            #self.cwt = np.real(self.cwt)


            self.scales*=PLOT_SR
            self.fUpdate['loma'] = True

        if self.fUpdate['tiers'] or self.fUpdate['cwt']:
            import matplotlib.colors as colors
            self.ax[-1].contourf(np.real(self.cwt),100,  norm=colors.SymLogNorm(linthresh=0.01, linscale=0.05,vmin=-1.0, vmax=1.0), cmap="jet")
            #self.ax[-1].contourf(self.cwt,100,  norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,vmin=-0.5, vmax=0.5), cmap="jet")

        # calculate lines of maximum and minimum amplitude
        if self.fUpdate['loma'] and labels:
            self.logger.debug("lines of maximum amplitude...")

            # get scale corresponding to avg unit length of selected tier
            unit_scale = misc.get_best_scale2(self.scales, labels)

            unit_scale = np.max([8,unit_scale])
            unit_scale = np.min([n_scales-2, unit_scale])
            labdur = []
            for l in labels:
                labdur.append(l[1]-l[0])


            # NOTE: scale numbers are somewhat arbitrary, should be possible to define by user

            pos_loma_start_scale = unit_scale - int(3./scale_dist) # three octaves down from average unit length
            pos_loma_end_scale = unit_scale
            neg_loma_start_scale = unit_scale - int(3./scale_dist)  # two octaves down
            neg_loma_end_scale = unit_scale + int(1./scale_dist)  # one octave up

            #some bug if starting from 0-3 scales
            pos_loma_start_scale = np.max([4, pos_loma_start_scale])
            neg_loma_start_scale = np.max([4, neg_loma_start_scale])
            pos_loma_end_scale = np.min([n_scales, pos_loma_end_scale])
            neg_loma_end_scale = np.min([n_scales, neg_loma_end_scale])

            pos_loma = loma.get_loma(np.real(self.cwt),self.scales,pos_loma_start_scale,pos_loma_end_scale) #,fig=self.ax[-1],color="black") #self.ax[2])
            loma.plot_loma(pos_loma, self.ax[-1], color="black")
            neg_loma = loma.get_loma(-np.real(self.cwt),self.scales,neg_loma_start_scale,neg_loma_end_scale) #,fig=self.ax[-1],color="white") #self.ax[2])
            loma.plot_loma(neg_loma, self.ax[-1], color="white")


            if labels:
                #max_loma= loma.get_max_per_label(pos_loma, labels)

                #max_loma_vals = np.array(max_loma)[:,1]
                max_loma = loma.get_prominences(pos_loma, labels)

                self.prominences=np.array(max_loma)
                #self.boundaries=np.array(loma._get_boundaries(max_loma, neg_loma, labels))
                self.boundaries=np.array(loma.get_boundaries(max_loma, neg_loma, labels))

            self.fUpdate['tiers'] = True

        # plot labels
        if self.fUpdate['tiers'] and labels:
            labels = self.tiers[unicode(self.tierlist.currentText())]
            text_prominence = self.prominences[:,1]/(np.max(self.prominences[:,1]))*2.5+0.5

            lab.plot_labels(labels,ypos=1, fig=self.ax[-1],size=5, prominences=text_prominence, boundary=True)

            for i in range(0,len(labels)):

                self.ax[-1].axvline(x=labels[i][1], color='black',linestyle="-",linewidth=self.boundaries[i][-1]*4,alpha=0.3)
                #self.ax[-1].axvline(x=labels[i][1], color='black',linestyle="-",linewidth=self.boundaries[i]*4,alpha=0.3)
                #self.ax[-1].axvline(x=self.boundaries[i][0], color='white',linestyle="-",linewidth=self.boundaries[i][-1]*4,alpha=0.3)
                #self.ax[-1].axvline(x=self.prominences[i][0], color='black',linestyle="-",linewidth=self.prominences[i][-1]*4,alpha=0.3)

        #
        # save analyses
        #
        #
        if labels:
            pass
            loma.save_analyses(os.path.splitext(unicode(self.cur_wav))[0]+".prom",labels, self.prominences, self.boundaries,PLOT_SR)


        #self.canvas.figure.subplots_adjust(wspace=None,hspace=None)
        self.ax[-1].set_ylim(0,n_scales)
        self.ax[-1].set_xlim(0,len(self.params))
        self.ax[0].set_ylabel("Spec (Hz)")
        self.ax[1].set_ylabel("F0 (Hz)")
        self.ax[2].set_ylabel("Signals")

        self.ax[2].set_yticklabels(["sum", "dur", "en", "f0"]) #"f0", "en", "dur", "sum"])
        self.ax[3].set_ylabel("Wavelet (scale)")

        plt.setp([a.get_xticklabels() for a in self.ax[0:-1]], visible=False)
        vals = self.ax[-1].get_xticks()[1:]
        ticks_x = ticker.FuncFormatter(lambda vals, p:'{:1.2f}'.format(float(vals/PLOT_SR)))
        self.ax[-1].xaxis.set_major_formatter(ticks_x)

        for i in range(0,4):
            nbins = len(self.ax[i].get_yticklabels())
            self.ax[i].yaxis.set_major_locator(MaxNLocator(nbins=5, prune='lower'))

        self.figure.subplots_adjust(hspace=0, wspace=0)

        if prev_zoom:
            self.ax[3].axis(prev_zoom)

        self.canvas.draw()
        self.canvas.show()



        self.fUpdate = dict.fromkeys(self.fUpdate, False)


##############################################################################################
## Main routine definition
##############################################################################################
def main():
    """Entry point which start the QT application
    """
    try:
        parser = argparse.ArgumentParser(description="")

        # Add options
        parser.add_argument("-v", "--verbosity", action="count", default=0,
                            help="increase output verbosity")

        # Add arguments
        # Example : parser.add_argument("echo", help="description")
        # TODO

        # Parsing arguments
        args = parser.parse_args()

        # Verbose level => logging level
        log_level = args.verbosity
        if (args.verbosity > len(LEVEL)):
            logging.warning("verbosity level is too high, I'm gonna assume you're taking the highes ")
            log_level = len(LEVEL) - 1
        logging.basicConfig(level=LEVEL[log_level])

        # Debug time
        start_time = time.time()
        logging.info("start time = " + time.asctime())

        # Running main function <=> run application
        app = QtWidgets.QApplication.instance()

        if not app:
            app = QtWidgets.QApplication(sys.argv)


        main = SigWindow()

        main.show()

        # Debug time
        logging.info("end time = " + time.asctime())
        logging.info('TOTAL TIME IN MINUTES: %02.2f' %
                     ((time.time() - start_time) / 60.0))

        # Exit program
        sys.exit(app.exec_())
    except KeyboardInterrupt as e:  # Ctrl-C
        raise e
    except SystemExit as e:  # sys.exit()
        pass
    except Exception as e:
        logging.error('ERROR, UNEXPECTED EXCEPTION')
        logging.error(str(e))
        traceback.print_exc(file=sys.stderr)
        sys.exit(-1)


###############################################################################
#  Envelopping
###############################################################################
if __name__ == '__main__':
    main()

# wavelet_gui.py ends here
