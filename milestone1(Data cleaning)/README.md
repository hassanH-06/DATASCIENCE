# Milestone 1

Start with **Milestone_1.html** to read the executed report.

- **Milestone_1.Rmd**: the submission source, with all R cleaning and EDA code embedded.
- **Milestone_1.html**: the rendered report with results and plots.
- **data/raw/framingham.csv**: the unchanged original dataset.
- **data/cleaned/framingham_cleaned.csv**: the cleaned dataset for inspection and Python handoff.
- **supporting/**: extra experiments, audit tables, figures, preprocessing state, and train/test handoff files.

To reproduce: open Milestone_1.Rmd in RStudio and click Knit. R packages rmarkdown
and knitr and Pandoc are required; RStudio supplies Pandoc. No separate analysis.R
is required. Keep supporting/ beside the Rmd because it contains the dictionary
and comparison evidence used by the report. Generated diagnostics are saved there.

Python cross-validation should start with supporting/handoff/train_unimputed.csv
and fit preprocessing within each training fold. Do not cross-validate the combined
already-imputed CSV. The report explains the exploratory holdout limitation and
the remaining source/license confirmation needed before submission.
