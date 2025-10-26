# Contributing

Thanks for your interest in improving **GWDroughtCalifornia**!

## How to contribute
- **Bug reports / questions:** open a GitHub Issue with a minimal reproducible example (snippet of the CSV and the console error).
- **Feature requests:** explain the decision need (e.g., export a shapefile; add kriging option; tweak color standards).
- **Pull requests:** fork → branch → PR. Please keep PRs focused, with clear before/after screenshots for figure changes.

## Code style
- Keep parameters near the top of the script (single source of truth).
- Avoid breaking the long-format input contract (`Well, LATITUDE, LONGITUDE, Date, Value`).
- Respect fixed DPI and stable color semantics (red = drier for anomalies/SGI; clearly documented legends).

## Tests / smoke checks
- Run a full end-to-end on the sample CSV and verify that all expected folders and files are created.

Thanks!
