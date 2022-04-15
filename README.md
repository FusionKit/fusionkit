# FusionKit
A framework of Python tools to enable the fusion community to perform fast, large scale, cross-code validation studies and dataset generation with gyrokinetic codes.

Current focus is on workflows that generate inputs for simulations with the GENE, QuaLiKiz and TGLF codes and fast analysis of large simulations datasets generated with these codes on remote systems. 

Future workflows will also be aimed at flux driven validation studies through integrated modelling (with i.a. JINTRAC) and support for more gyrokinetics codes will be added.

## Getting started
To use FusionKit in a project get started by:
```bash 
$ git clone git@github.com:FusionKit/fusionkit.git
$ cd fusionkit
$ pip install --user -e .
```
Join the repository to contribute or raise issues!

## Framework structure
FusionKit consists of a small core that handles all data and operations related to the plasma state to be simulated and extensions that handle the interfacing with external codes.
### Core Classes
- **DataSpine**, a class to handle reading and writing different class objects to/from disk.
- **Plasma**, a class to handle all data (experimental/simulation) related to a discharge time-slice for input into a simulation, currently includes:
    - handling species data self-consistently,
    - automatic remapping of all species data on an Equilibrium.
- **Equilibrium**, a class to handle all equilibrium related mapping and calculation of geometry derived quantities, currently includes:
    - g-eqdsk reader & writer, 
    - flux surface tracing functionality,
    - derivation of local geometry quantities for s-alpha and Miller (MXH, Fourier to be added).
  
### Extensions
- **EX2GK** (incl. tools for reading EX2GK output, time-averaging and filtering tools to use in custom filtering of experimental data in EX2GK)
- **GENE** (incl. tools for generating GENE parameter files, setting-up runs on and collecting results from a remote system)
- **JET_PPF** (incl. tools for reading PPF output from JETDSP exports)
- **TGLF** (incl. tools for generating TGLF input files)
- **QLK** (incl. tools for generating QuaLiKiz input files and setting up local runs)

Development is ongoing, so more functionality is added on a daily basis.

## Users & developers
- DIFFER
- EX2GK @ CCFE
- Aix-Marseille Université
