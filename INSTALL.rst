Install procedure
=================

Wavelet Prosody Analyzer is a toolkit comprising command line tools and a GUI application.
All the tools are started from terminal, so some familiarity with command line tools is assumed.

Installation has been tested only on one Ubuntu Linux, on Arch Linux and on MacOS Sierra machine.
Running on windows might be possible if the required libraries can be installed.

Default installation
---------------------

To install the toolkit, simply run

.. code:: sh

    pip install -e .

It will install the dependencies needed to run the toolkit.

To be able to run the application globally, the following line should be added to your shell profile file (~/.bashrc or ~/.profile in general):

.. code:: sh

   export PATH=~/.local/bin:$PATH

After restarting the shell, you can finally run the tool by calling them on the command line, like for example:

.. code:: sh

   wavelet_gui

Development mode installation
------------------------

Even if the setup doesn't require it, we advise to use the environment management system conda ( https://docs.conda.io/en/latest/miniconda.html ).
Conda provides an easy way to define the environments and install precompiled packages.
Therefore, the modification you will propose won't affect your system configuration.

Assuming you have created activated the conda environment, you can install pre-compiled packages

.. code:: sh

   conda install scipy numpy matplotlib joblib pyqt

We then use the setup script to install the rest of the dependencies:

.. code:: sh

    pip install -e .

To start the Wavelet Prosody Analyzer GUI, run the following commands:

.. code:: sh

    python3 wavelet_prosody_toolkit/wavelet_gui.py

if it doesn’t work, please raise an issue on github here: https://github.com/asuni/wavelet_prosody_toolkit/issues .
