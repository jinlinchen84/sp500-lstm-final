# Constituent reconstruction note

- Input is month-end S&P 500 membership snapshots.
- Each month-end snapshot is shifted forward to represent membership in the subsequent month.
- Output is an effective-date table with `effective_start` and `effective_end`.
- This produces a monthly binary membership matrix indicating whether a stock belongs to the index in month t+1.
- This monthly membership is later used to define the stock universe at the end of each training window for rolling study periods.
