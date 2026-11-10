### qudit-cut

Circuit cutting for qudit quantum circuits: cuts of the controlled-sum (CSUM) gate between qudits of any two dimensions.

<img src="local/logo.svg" width="140" alt="mark">

[`local/logo.svg`](local/logo.svg): a recreation of the Heart Pirates' Jolly Roger (ring, six spoke-and-T
marks, split half outline / half solid), unrelated to the above.

#### Paper

`local/report_v2.tex` with bibliography `local/ref.bib` and writing rules `local/style.md`. Build with [tectonic](https://tectonic-typesetting.github.io):

```
cd local && tectonic report_v2.tex
```

#### Numerics

`scripts/hw_optimal_cut.py` simulates the LO cut and the LOCC cut on random circuits and prints the overhead and the largest total variation distance per pair of dimensions (the table in the paper). It uses the [qudit](https://ieeexplore.ieee.org/document/11333840) simulator:

```
uv run --no-project --with qudit python scripts/hw_optimal_cut.py
```
