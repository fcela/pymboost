# Warm In-Process Benchmark Results

| Case | Python mean (s) | R mean (s) | Python/R speedup |
|---|---:|---:|---:|
| `glmboost_gaussian_bols` | 0.001722 | 0.007192 | 4.18x |
| `glmboost_binomial_bols` | 0.002420 | 0.014220 | 5.88x |
| `glmboost_poisson_bols` | 0.001907 | 0.008789 | 4.61x |
| `gamboost_gaussian_bbs_bols` | 0.006936 | 0.020010 | 2.89x |
| `gamboost_gaussian_bmono` | 0.067572 | 0.036580 | 0.54x |
| `gamboost_gaussian_btree` | 0.043395 | 0.092202 | 2.12x |
| `cvrisk_gaussian_bols` | 0.004185 | 0.178415 | 42.64x |
| `cvrisk_gaussian_bmono` | 0.072833 | 0.141196 | 1.94x |
| `cvrisk_gaussian_btree` | 0.110702 | 0.298336 | 2.69x |
