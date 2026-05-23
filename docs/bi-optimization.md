## BI Optimization

`prelim.sd.bi` now points to the optimized BI implementation.

The previous implementation is preserved as:

- [bi_slow.py](/C:/Users/arzamasov/OneDrive/Documents/KIT_PRojects_to_Finish/2022_1_PRELIM/prelim/src/prelim/sd/bi_slow.py)

The optimized implementation lives at:

- [bi.py](/C:/Users/arzamasov/OneDrive/Documents/KIT_PRojects_to_Finish/2022_1_PRELIM/prelim/src/prelim/sd/bi.py)

### What changed

- `_refine` was rewritten to avoid repeated masked rescans over the same column values.
- The optimized version uses grouped counts with prefix/suffix sums, which removes the main quadratic hot path from the original implementation.
- Beam pruning preserves the original tie-breaking semantics, so the optimized implementation matches the old BI outputs in the comparison run.

### Compatibility check

The comparison was run over all experiment datasets with:

- HPO on `100` rows
- final fit on up to `10000` rows
- identical preprocessing for old and new BI

Final result:

- `30/30` datasets matched on score
- `30/30` datasets matched on final box
- `30/30` datasets matched on refit depth

Runtime on the final fit only:

- old total: `903.70s`
- new total: `34.61s`
- aggregate speedup: about `26.1x`

The generated comparison table is:

- [bi_comparison_10000.csv](/C:/Users/arzamasov/OneDrive/Documents/KIT_PRojects_to_Finish/2022_1_PRELIM/prelim/docs/assets/bi_comparison_10000.csv)
