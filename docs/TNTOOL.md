# Independent TNTool validation

The repository does **not** redistribute Plamen Koev's TNTool. The factor data
and an interoperability runner are supplied so the external comparison can be
repeated with an independently obtained TNTool installation.

Official TNTool page:

https://sites.google.com/sjsu.edu/plamenkoev/home/software/tntool

After adding the TNTool directory to the MATLAB or Octave path, run from the
repository root:

```matlab
run('external/run_tntool_comparison.m')
```

The runner reads:

- `data/koev_tntool_bd.csv`
- `data/koev_tntool_reference.csv`

and writes `results/koev_tntool_comparison.csv`.

The repository also retains the independently measured comparison used in the
retained study results. `scripts/summarize_external_tntool.py` checks and summarizes that
stored table without claiming to execute TNTool locally.
