# How to generate the documentation

- extract the info from the source code
```sh
sphinx-apidoc ../wavelet_prosody_toolkit -o _modules -e -M
```
- generate the html documentation
```sh
make html
```
- documentation is generated in `../build/docs/html`
