# Laptop Price Lab

A responsive website for the four frozen laptop-price models: linear regression, ridge regression, random forest and XGBoost. Enter laptop specifications or load a real held-out test example. Compare predictions, actual listed prices and absolute errors across price ranges.

## Run locally

Run `python -m http.server 4173 --directory dist` and open http://localhost:4173. No Python inference server, API key or training is needed. The browser runs the exported fitted model parameters.

## Files

- `dist/index.html`, `style.css`, `app.mjs`: website and interaction code.
- `dist/engine.mjs`: deterministic feature engineering, saved preprocessing and model inference.
- `dist/data/`: exported frozen models, training vocabularies, real test examples and provenance.
- `scripts/export_models.py`: exports the existing saved Python pipelines; never retrains. Pass the original Milestone 2 folder as its argument.
- `scripts/verify.mjs`, `parity_cases.json`: compare browser inference with all four original pipelines on all 432 test rows. Run `node scripts/verify.mjs`.
- `.openai/hosting.json`: private Sites hosting configuration.

## Input and evaluation behavior

Training-vocabulary dropdowns preserve the expected text formats. CPU family is filled from the saved CPU-model mapping. Four CPU/GPU summary columns are derived automatically. Numeric imputation/scaling, categorical one-hot encoding (including rare-category grouping) and log-price inversion preserve the frozen pipelines. Tree inference uses the original fitted thresholds and parameters; models are not trained or tuned by the website.

Actual price is never used as an input. It is shown only when an unchanged selected test example matches the submitted specifications. Editing an example removes that comparison until its specifications match again. Custom inputs show predictions without inventing an actual price. Prices use source dataset units, because currency has not been verified. Inference runs locally in the browser, so form inputs are not sent to a prediction service.

Examples are fixed price-ranked selections from the existing 432-row project test set, not new independent evaluation data. The interface is for demonstration; browsing examples does not update the models. Coverage is limited to the original dataset, and expensive or uncommon configurations may have larger errors. These are price estimates, not current market quotes or prediction intervals.

Team: Mostafa Bdeir, Hassan Hijazi and Hasan Haidar.

## Deploy on Netlify

Import `hassanH-06/DATASCIENCE` from GitHub, select the `main` branch, leave the build command empty and publish `dist`. The included `netlify.toml` supplies the publish directory. No backend or environment variables are needed. The default Netlify URL does not contain ChatGPT.
