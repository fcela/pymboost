```{code-cell} ipython3
bodyfat = pl.read_csv(ROOT / "data" / "bodyfat.csv")
r_assign_dataframe("bodyfat_py", bodyfat.to_pandas())
ro.r("bodyfat <- bodyfat_py")
predictor_names = [name for name in bodyfat.columns if name != "DEXfat"]
bodyfat
```
