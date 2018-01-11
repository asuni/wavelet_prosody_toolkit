# Install procedure

Wavelet Prosody Analyzer is a graphical tool built with Python3 and QT5.
It is started from terminal, so some familiarity with command line tools is assumed.

Installation has been tested only on one Ubuntu Linux, on Arch Linux and on MacOS Sierra machine.
Running on windows might be possible if the required libraries can be installed.

## Standard installation

Using the setup installation script, you can install it locally using the following command:
```sh
python3 setup.py install --user
```

or for all users, **with root privileges**, using this command:
```sh
python3 setup.py install
```

## Developer installation

- When using mac, check if command line tools have been installed. In terminal, print:
```sh
gcc --version
```
if you get a pop-up box asking about installing, press 'Install'

- Check if python3 is available:
```sh
python3 --version
```
if you get "command not found" or version < 3.5, download python3.6 from python.org and install.

- Verify the install:
```sh
python3 --version
```
> python3 --version
should print something like: "Python 3.6.3 (v3.6.3:2c5fed86e0, Oct  3 2017, 00:32:08)"

- install required python libraries:
```sh
python3 --version
```

- cd to the directory where the Wavelet Prosody Analyzer was extracted and start prosody analyzer:
> python3 wavelet_gui.py

if it doesn't work, contact antti.suni@helsinki.fi
