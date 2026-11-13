# Bank of Canada zero-coupon yield curve

This folder is for one small CSV that you download yourself. The repository never
creates or edits curve data.

## What to download

1. Open https://www.bankofcanada.ca/rates/interest-rates/bond-yield-curves/
2. Under **Yield Curve Data**, choose **"Retrieve data for the following period"**.
3. Enter **one date** as both the start and the end date. The data is updated on
   Thursdays with a two-week lag, so pick a date at least two weeks ago.
   Do **not** use "Retrieve all data": that returns every curve since 1986.
4. Save the file here, e.g. `data/boc/zc_2026-10-29.csv`. It should be a few KB.

Checked on 2026-11-13, the page says:
- each row is one zero-coupon curve;
- values are decimals (0.0500 = 5.00%);
- terms run from 0.25 years (column 1) to 30.00 years (column 120);
- missing values are shown as `na`.

## How to use it

In a config:

```yaml
economy:
  curve:
    source: boc_csv
    boc_csv: data/boc/zc_2026-10-29.csv
    boc_date: "2026-10-29"
```

The loader (`segfund.curves.load_boc_zero_curve`) refuses to guess. It raises an
error if a row doesn't have 120 yield fields or if the values look like percent.

**Not yet verified:** the loader was written from the page description, before a
real file was inspected. It also assumes continuous compounding. Both have to be
checked against a real download (and the BoC methodology note) before any number
based on the BoC curve is reported.
