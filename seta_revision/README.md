# SETA revision package

This directory contains the current revision analysis code and selected
aggregate tables. It is deliberately separated from the older workflow kept
at the repository root.

The analysis scripts use an authorized local source project through the
`SETA_PROJECT_ROOT` environment variable. No purchased or provider-controlled
input data are copied into this directory. Detailed station rows, event rows,
timestamp masks, raw fields, and local metadata are excluded.

`data/aggregate/` contains only compact summaries selected for public release.
They are provided for result inspection and citation support; they are not a
replacement for the restricted source data and do not make the raw pipeline
self-contained.

The current scripts are syntax-checked in the release workflow. A full
scientific rerun requires the separately obtained input files and their exact
provider versions.
