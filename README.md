# Supplementary material: Expressing general constitutive models in FEniCSx using external operators and algorithmic automatic differentiation

## Links

- isSupplementedBy *software* [https://doi.org/10.5281/zenodo.10907417](https://doi.org/10.5281/zenodo.10907417)
- isSupplementTo *publication-article* [https://doi.org/10.46298/jtcam.14449](https://doi.org/10.46298/jtcam.14449)
- isSupplementTo *publication-article* [https://hal.science/hal-04735022v2](https://hal.science/hal-04735022v2)
- isSupplementedBy *software* [https://github.com/a-latyshev/dolfinx-external-operator/tree/v0.9.0](https://github.com/a-latyshev/dolfinx-external-operator/tree/v0.9.0)
- TODO v4: isSupplementedBy *software* [https://doi.org/10.5281/zenodo.13908686](https://doi.org/10.5281/zenodo.13908686)

## Authors

- **Latyshev, Andrey**, University of Luxembourg, ORCID: [0009-0002-7512-0413](https://orcid.org/0009-0002-7512-0413)

## Language

- English

## License

- Creative Commons Attribution 4.0
- LGPL v3.0 or later

Contributions
-------------

* Latyshev, Andrey: Data collection, Visualisation.

Data collection: period and details
-----------------------------------

From 10/2024 to 01/2025.

Funding sources
---------------

This research was funded in whole, or in part, by the Luxembourg National Research
Fund (FNR), grant reference PRIDE/21/16747448/MATHCODA. For the purpose of open access,
and in fulfilment of the obligations arising from the grant agreement, the author has applied a
Creative Commons Attribution 4.0 International (CC BY 4.0) license to any Author Accepted
Manuscript version arising from this submission.

Data structure and information
------------------------------

Supplementary material for the scientific paper "Expressing general
constitutive models in FEniCSx using external operators and algorithmic
automatic differentiation". It includes a Python script generating the plots of
the paper from their data. 


JTCAM curation policy
---------------------

**First things first: JTCAM data editors are here to help!**

**Second: the JTCAM data curation policy** can be found here: https://zenodo.org/communities/jtcam/curation-policy

JTCAM editors wish to follow the FAIR principles regarding the open data and algorithms: https://en.wikipedia.org/wiki/FAIR_data

The intention is to publish data relevant to the JTCAM paper. This includes:

- Raw data used to produce Figures
- Experimental data
- Simulation Input and Output
- Algorithms (software, scripts)

In essence, this *Open Data* approach will allow **reproducibility** and an **easy analysis by another research group**.

Data structure and information
------------------------------

+ Folder/files structure:
  + `README.md` - Describes the contents and structure of the supplementary materials.
  + `supplementary_materials` - Main folder containing supplementary material.
    + `plots` - Scripts and raw data to make plots shown in paper.
      + `data` - Contains raw data used in plots.
        + `k_list.npy` - NumPy array with values of scaling parameter for the Taylor test. **Used in Figure 4.**
        + `performance_data_200x200_n_{n}.pkl` - Python dictionaries containing timings breakdown including total time, matrix assembly, linear solver, vector assembly, constitutive model update and loading step and Newton iteration indexes. The data is stored via the `pickle` Python module for n MPI processes (n = 1, 2, 4, 8, 16, 32, 64). Generated using `../scaling/demo_plasticity_mohr_coloumb_mpi.py`. **Used in Figures 7 and 8 (Appendix A.1).**
        + `results_mohr_coulomb_non_associative.npy` -  NumPy array with displacement and soil self-weight values for the Mohr-Coulomb problem (non-associative flow). **Used in Figure 6.**
        + `results_mohr_coulomb.npy` - NumPy array with displacement and soil self-weight values for the Mohr-Coulomb problem (associative flow). **Used in Figure 6.**
        + `results_von_mises_pure_ufl.npy` - NumPy array with displacement and applied pressure values for the von Mises problem (pure UFL). **Used in Figure 2.**
        + `results_von_mises.npy` - NumPy array with displacement and applied pressure values for the von Mises problem (Numba/external operator). **Used in Figure 2.**
        + `rho_returned.npy` - NumPy array with with radial coordinates restored after return-mapping for the Mohr-Coulomb yield surface. **Used in Figure 3.**
        + `rho_standard_MC.npy` - NumPy array with radial coordinates for the standard Mohr-Coulomb yield surface (analytical).  **Used in Figure 3.**
        + `slope_displacement.npy` -  NumPy array with values of displacements in mesh nodes to compute the magnitude of the slope slip. **Used in Figure 5.**
        + `slope.npy` - NumPy array representing image of the deformed slope generated via pyvista. **Used in Figure 5.**
        + `taylor_reminders_data.npy` - NumPy array with Taylor remainder norms for the Taylor test. **Used in Figure 4.**
        + `theta_returned.npy` - NumPy array with Lode angles restored after return-mapping for the Mohr-Coulomb yield surface. **Used in Figure 3.**
        + `theta_values.npy` - NumPy array with Lode angles for the standard Mohr-Coulomb yield surface (analytical). **Used in Figure 3.**
      + `docker/Dockerfile` - Dockerfile containing description of environment
        used to generate plots from the paper. The built image (x86-64) is
        available on Zenodo.
      + `output/` - Empty directory for output of `demo_plasticity_mohr_coulomb_mpi.py`.
      + `plots_for_papers.py` - Python script that plots all of the figures from the paper from the data in `data/`.
      + `README.md` - Further instructions.
    + `scaling/` - Folder containing scripts for the strong scaling performance test, for different MPI process counts **(Appendix A.1)**.
      + `docker/Dockerfile` - Dockerfile containing description of environment used to execute the strong scaling test. The built image (x86-64) is available on Zenodo.
      + `output/` - Empty directory for output of `demo_plasticity_mohr_coulomb_mpi.py`.
      + `demo_plasticity_mohr_coulomb_mpi.py` - Python script for the Mohr-Coulomb plasticity problem. Generates data for the strong scaling test **(Appendix A.1)**.
      + `README.md` - Further instructions.
      + `solvers.py` - Additional methods used in `demo_plasticity_mohr_coulomb_mpi.py`.
      + `utilities.py` - Additional methods used in `demo_plasticity_mohr_coulomb_mpi.py`.

Paper Description
-----------------

Many problems in solid mechanics involve general and non-trivial constitutive
models that are difficult to express in variational form. Consequently, it can
be challenging to express these problems in automated finite element solvers,
such as the FEniCS Project, that use domain-specific languages specifically
designed for writing variational forms. In this article, we describe a
methodology and software framework for FEniCSx / DOLFINx that enables the
expression of constitutive models in nearly any general programming language.
We demonstrate our approach on two solid mechanics problems; the first is a
simple von Mises elastoplastic model with isotropic hardening implemented with
Numba, and the second a more complex Mohr-Coulomb elastoplastic model with apex
smoothing implemented with JAX. In the latter case we show that by leveraging
JAX's algorithmic automatic differentiation transformations we can avoid
error-prone manual differentiation of the terms necessary to resolve the
constitutive model. We show extensive numerical results, including Taylor
remainder testing, that verify the correctness of our implementation. The
software framework and fully documented examples are available as supplementary
material under the LGPLv3 or later license.

Supplementary material for the scientific paper "Expressing general
constitutive models in FEniCSx using external operators and algorithmic
automatic differentiation". It includes a Python script generating the plots of
the paper from their data.
