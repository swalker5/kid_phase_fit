# `kid_phase_fit`

## Project structure

This code is structured so that `kid_phase_fit.py` will take a microwave power sweep of microwave kinetic inductance detector (MKID) data (currently in .nc format for now) and fit its phase. The code requires at least two sweeps, one low power dataset and another dataset at any microwave power.

## Build

```
    $ git clone /url/to/this/repo
```

## Code Demo

```
    $ python kid_fit_phase.py config.yaml
```

## Notes

### Regarding `config.yaml`
The file `config.yaml` contains user-specified parameters to be used by `kid_phase_fit.py`, separated into different sections. The `load` section corresponds to the folders to find data and save the code output. The variable `sweep_name` corresponds to the beginning of a particular network and obsnum. The `save` section corresponds to how to save the data fit, as figures, a PDF, or in a .pkl file. The `preview` section corresponds to showing plots while running the code or not. The `weight` section corresponds using a weighting around a tone frequency and quality factor or not. The `flag_settings` section corresponds to guessing the optimal drive power for each resonator corresponding to a user-defined nonlinearity value as well as thresholds for the goodness of fit, in particular the residuals of the phase fit, and nonlinearity parameter guess for each fit resonator. The `fit_settings` section corresponds to setting the microwave power range to fit, how many resonator linewidths to fit, and which resonators to fit.

### Regarding csv file beginning with `drive_atten`
The csv file contains various information about the fits after running `kid_phase_fit.py`. The variable `tone_num` specifies the tone/resonator. The variable `drive_atten` specifies the drive attenuation guessed after fitting the power sweep for a user-specified nonlinearity parameter for the network in `config.yaml`, in particular the variable `a_predict_guess`. The `drive_atten_flag` corresponds to the whether the `a_predict_guess` is above a user-specified threshold in `config.yaml`, in particular the variable `a_predict_threshold`. For `drive_atten_flag`, a value of 0 is a good guess/below the threshold and a value of 1 is above the threshold. The variable `fit_success` (same length as the microwave power sweep) corresponds to whether each fit for one power failed or not, 0 for a pass, 1 for a fit that was skipped. The variable `fit_flags` (same length as the microwave power sweep) corresponds to the goodness of the fit, in particular the residuals of the phase fit, compared to a threshold as specified by the variables `pherr_threshold` (residual threshold) and `pherr_threshold_num` (number of points above threshold) in `config.yaml`.

The flag settings are as follows:

0 = good fit, below user-defined thresholds

1 = failed fit 

2 = above user-defined thresholds

Also, an empty list corresponds to a skipped fit/likely not a resonator.
