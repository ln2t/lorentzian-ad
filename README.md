# Lorentzian Parameterization of MEG Power Spectra in Alzheimer’s Disease

## Overview

This repository implements and motivates a **Lorentzian parameterization of MEG power spectra** designed to study neurophysiological alterations in Alzheimer’s disease (AD).
It's ambition is not to give a full framework, but more as a proposition to study. It is partially derived from [FOOOF/Specparam](https://fooof-tools.github.io/fooof/), it is focused on the Lorentzian parameterization of the power spectrum, it has been developed for the following data: https://osf.io/t753c/.

## Why a Lorentzian model?

A Lorentzian spectrum emerges naturally from first order filters.

$$
\text{logPSD}(f) = A - \text{log}(1+(\frac{f}{f_c})^{-B})
$$

where:

-   $A$ reflects the gain
-   $f_c$ is a cutoff frequency linked to the **synaptic time constant**
-   $B$ controls the approximate power-law decay at high frequencies ($f >> f_c$)

## Implementation details

![Image](./illustration.png)

1. Fit a first <span style="color:#D14B4B;"><b>Lorentzian</b></span> to 4–50 Hz, remove the segments above the fit that contain a peak, and fit a second <span style="color:#D14B4B;"><b>Lorentzian</b></span>
2. Fit peaks by taking the highest point in the residuals and iteratively fitting a Gaussian model until a stopping threshold (as in [FOOOF/Specparam](https://fooof-tools.github.io/fooof/))
3. Fit a <span style="color:#5BD14B;"><b>power-law</b></span> below 4 Hz in log–log space, on the residuals of the Lorentzian (robust Tukey fit)
4. Fit the <span style="color:#3E6DDA;"><b>neural excitability index</b></span>, a power-law over 15–50 Hz, in two passes (Gao et al. 2017)

The 49.8–50.2 Hz bins are removed before the fits. $B$ (`l_exp`) and the low-frequency exponent (`pl_exp`) are signed log-log slopes.

## Reproducing the paper

Put `fdg_data.csv` and the decompressed `lorentzian_ad_data/` folder in `data/`, then run `main.ipynb`: spectral fits, then the statistics of the paper.
