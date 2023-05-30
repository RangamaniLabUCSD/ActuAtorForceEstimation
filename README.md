# Automembrane
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/RangamaniLabUCSD/ActuAtorForceEstimation/workflows/CI/badge.svg)](https://github.com/RangamaniLabUCSD/ActuAtorForceEstimation/actions?query=workflow%3ACI)
## Estimation of forces induced on the membrane by ActuAtor

This package contains a tool `automembrane` which employs automatic differentiation to compute the energy and forces of the membrane given a 2D discrete shape profile.
We apply the tool to a dataset where ActuAtor is used to deform the nuclear membrane.
A [preprint](https://www.biorxiv.org/content/10.1101/2020.03.30.016360) of this work is available on the bioRxiv.

## Local directory structure
```txt
ActuAtorForceEstimation
├── actuator_example    ->  Code for application to ActuAtor
│   ├── coordinates     ->  Initial segmentations
│   ├── crop_images     ->  Cropped images
│   └── raw_images      ->  Raw EM images
├── automembrane        ->  Source for automembrane tool
│   └── tests
├── devtools
│   └── conda-envs
└── examples
```

## Installing automembrane and running the example

After cloning this repository, `automembrane` can be installed by running `pip install .` from the root folder.

Examples can be found in the `actuator_example` folder. 
The raw/cropped EM images and segmentations are in the corresponding subfolders.
There are several python files which perform the analysis and work.

```txt
actuator_constants.py   -> Contains mappings for file locations and EM image metadata
make_movie.py           -> Helper for making movies

main_sweep.py       -> Main driver function which runs relaxation, parameter variation and plotting

The following should be run in order:
variation_relax.py  -> Code for relaxation of initial segmentented geometries
variation_run.py    -> Runs parameter variation on the relaxed geometries
variation_view.py   -> Generated plots and movies
```

If you are short on time, you can simply run `main_sweep.py` which will perform geometric relaxation, parameter variation, and plotting for you automatically.
It calls functions from the three `variations_*.py` files which do the actual work.

For a guided approach, there are also two jupyter notebooks which discuss the theory behind the work.
You can also execute these to get a feel for what is happening in the automatic scripts.

## Copyright

Copyright (c) 2022-2023, Eleanor Jung, Cuncheng Zhu, Christopher T. Lee, and Padmini Rangamani

### Acknowledgements

Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
