# Gaze Bias Differences Capture Individual Choice Behavior

This repository contains data and python code to reproduce all analyses performed in

**Gaze Bias Differences Capture Individual Choice Behavior**  
Thomas, A. W.* & Molter, F.*, Krajbich, I., Heekeren, H. R. H. & Mohr, P. N. C.  
Nature Human Behaviour, 2019, X(X), p. XXX.  
doi: http://dx.doi.org/10.1038/s41562-019-0584-8

*shared first authorship with equal contribution

The main analyses can be followed within multiple Jupyter notebooks (files with .ipynb extension):

- `0_data_preprocessing.ipynb`: Preprocessing of the four included datasets
- `1_individual_differences.ipynb`: Descriptive results and basic behavioural analyses
- `2_relative_model_fit.ipynb`: Evaluation of results from within-subject model comparison
    - Model fitting and comparison performed using the `GLAM_insample_comparison.py` script
- `3_absolute_model_fit.ipynb`: Evaluation of absolute model fit (out of sample prediction)
    - Model fitting and prediction performed using the `GLAM_oos_prediction.py` script
- `4_glam_parameters_predict_behaviour.ipynb`: Analysis of relationships between model parameters and behavioural measures

Additional supplementary analyses are contained in the following Jupyter notebooks:

- `SI_0_convergence_check.ipynb`: Convergence checks for MCMC traces
- `SI_1_parameter_estimates.ipynb`: Visualization of parameter estimates (Supplementary Figure 1)
- `SI_2_multiplicative_vs_additive.ipynb`: Individual comparison between multiplicative and additive GLAM variants (Supplementary Figure 2)
- `SI_3_additive-vs-multiplicative_group-averaged.ipynb`: Group comparison between multiplicative and additive GLAM variants (Supplementary Figure 3)
- `SI_4_OOS_predicted_behavioural_metrics.ipynb`: Visualization of out-of-sample predicted individual differences and relations on behavioural metrics (Supplementary Figure 4)
- `SI_5_6_Individual_RT_distributions.ipynb`: Visualization of group and individual response time distributions (Supplementary Figures 5 and 6)
- `SI_7_parameter_recovery.ipynb`: Visualization of parameter recovery analysis
    - Recovery performed using `GLAM_parameter_recovery.py` script

The files `analysis_functions.py` and `plotting_functions.py` contain shared functions that are loaded by each notebook separately.
