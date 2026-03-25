# Model Artifacts

Training and evaluation outputs are written here by `backend/train.py` and retraining flows in `backend/main.py`.

## Typical files

- `fake_news_model.joblib`
- `training_splits.joblib`
- `model_metrics.json`
- generated plot images

## Notes

- These files are runtime outputs of training rather than source code
- Some plot images may be committed for presentation purposes
- Large generated artifacts are generally treated as local outputs
- If `fake_news_model.joblib` is missing, the backend falls back to heuristic demo predictions instead of failing to start
