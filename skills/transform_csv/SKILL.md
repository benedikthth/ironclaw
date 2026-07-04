---
name: transform_csv
description: Read a CSV, aggregate a numeric column by a key column, write a summary.
---

# Transform CSV

A reusable recipe for the common "collect then summarize tabular data" step.

## Procedure

1. Read the input CSV with the stdlib `csv` module (no third-party deps needed).
2. Group rows by the requested key column; sum the requested value column,
   coercing values with `float()`.
3. Write the result as a two-column CSV (`key,total`), rows sorted by key, so the
   output is itself queryable by downstream tasks.

## Reference implementation

```python
import csv, collections
totals = collections.defaultdict(float)
with open(SRC, newline="") as fh:
    for row in csv.DictReader(fh):
        totals[row[KEY]] += float(row[VALUE])
with open(DST, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["key", "total"])
    for k in sorted(totals):
        w.writerow([k, totals[k]])
```
