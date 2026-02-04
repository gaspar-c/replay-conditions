# Replay Conditions — Figure Scripts

Each `fig*.py` script generates one or more figure panels. Command-line usage (including optional arguments) is documented in each script's header.

## Reproducibility
Use `environment.yml` to recreate the conda environment used for the results in the paper.

## Quick Reference

- `fig1BC.py`: Generates Figure 1, panels B and C for models 1–3. Prompts to select a model.
- `fig1D.py`: Generates Figure 1, panel D[^1][^2] for models 1–3. Prompts to select a model. Also supports plot-only via `python fig1D.py plot`.
- `fig2A.py`: Generates Figure 2, panel A[^1][^2]. Also supports plot-only via `python fig2A.py plot`.
- `fig2B.py`: Generates Figure 2, panel B[^1]. Also supports plot-only via `python fig2B.py plot`.
- `fig4AC.py`: Generates Figure 4, panels A and C. Prompts to choose model (`gaussian` or `rectangle`). Also supports plot-only via `python fig4AC.py plot`.
- `fig4BD.py`: Generates Figure 4, panels B and D. Prompts to choose model (`gaussian` or `rectangle`).
- `fig4E.py`: Generates Figure 4, panel E. Also supports plot-only via `python fig4E.py plot`.
- `fig4F.py`: Generates Figure 4, panel F.
- `figS2AB.py`: Generates Figure S2, panels A and B.
- `figS2C.py`: Generates Figure S2, panel C[^1][^2]. Also supports plot-only via `python figS2C.py plot`.
- `figS2D.py`: Generates Figure S2, panel D. Also supports plot-only via `python figS2D.py plot`.
- `figS3AB.py`: Generates Figure S3, panels A and B.
- `figS3C.py`: Generates Figure S3, panel C[^1][^2]. Also supports plot-only via `python figS3C.py plot`.


Outputs are written to the `outputs/` directory with timestamped subfolders created per run.

## Repository Structure

```
replay-conditions/
├─ README.md
├─ environment.yml
├─ fig1BC.py
├─ fig1D.py
├─ fig2A.py
├─ fig2B.py
├─ fig4AC.py
├─ fig4BD.py
├─ fig4E.py
├─ fig4F.py
├─ figS2AB.py
├─ figS2C.py
├─ figS2D.py
├─ figS3AB.py
├─ figS3C.py
├─ general_code/
│  ├─ aux_functions.py
│  ├─ group_simulations.py
│  └─ parameters.py
├─ population_model/
│  ├─ group_replay_analysis.py
│  ├─ model_gauss.py
│  ├─ model_rectangle.py
│  ├─ simulations.py
│  └─ theory.py
├─ spiking_networks/
│  ├─ connectivities.py
│  ├─ group_replay_analysis.py
│  ├─ model1.py
│  ├─ model2.py
│  ├─ model3.py
│  ├─ model3_delta.py
│  ├─ model3_no_leak.py
│  ├─ network.py
│  ├─ plot_spiking_trace.py
│  ├─ simulations.py
│  ├─ synapses.py
│  └─ tests.py
├─ imported_code/
│  └─ detect_peaks.py
└─ outputs/
```

[^1]: For quick runs, these scripts default to `conn_seeds = [1]`. Manuscript results used `conn_seeds = [1, 2, 3, 4, 5]`; 5 pseudo-random instantiations of each (p_ff, p_rc) parameter pair.

[^2]: For quick runs, these scripts default to `num_stims = 1` . Manuscript results used `num_stims = 5`; 5 replay trials on each simulated network.
