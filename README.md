# Chicken Disease Classification

Lightweight project that builds and trains a CNN to classify chicken health from fecal images (healthy vs diseased).  
Implemented as a reproducible pipeline with separate stages for data ingestion, preparing a base model, training, and evaluation.

## Features
- Data ingestion from a hosted zip archive
- Prepare base model (transfer learning) and save updated base model
- Training pipeline with TensorBoard and checkpoint callbacks
- Evaluation pipeline producing a JSON score file
- Configuration-driven (YAML) for easy experiments

## Requirements
- Python 3.8+
- Recommended: conda environment (example name: `chicken`)
- GPU optional (TensorFlow will use available devices)

Install dependencies:
```bash
conda create -n chicken python=3.8 -y
conda activate chicken
pip install -r requirements.txt
```

## Project layout
- src/chicken_disease_prediction/  
  - components/         # training, evaluation, data ingestion components
  - config/             # configuration management
  - pipeline/           # pipeline stage orchestrations (stage_01..stage_04)
  - entity/             # dataclasses for configuration entities
  - utils/              # helpers (yaml, dirs, logging, etc.)
  - constants.py        # CONFIG_FILE_PATH, PARAMS_FILE_PATH, etc.
- config/
  - config.yaml         # pipeline configuration (paths, filenames)
  - params.yaml         # hyperparameters (epochs, batch size, image size)
  - secrets.yaml        # (optional) any secrets
- artifacts/             # generated outputs (models, logs, tensorboard, checkpoints)
- main.py                # entrypoint to run the full pipeline
- README.md

## Quick start
From project root:
```bash
# set working directory to project root (if using notebooks)
# run full pipeline (data ingestion -> prepare base model -> training -> evaluation)
python main.py
```

## Configuration
Edit `config/config.yaml` and `config/params.yaml` to change paths and hyperparameters:
- config.yaml controls artifact locations, data unzip dir, checkpoint & tensorboard paths
- params.yaml controls IMAGE_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, AUGMENTATION, etc.

After edits restart the process so the ConfigurationManager reads updated YAMLs.

## Running stages individually
You can run stage scripts directly (see `src/chicken_disease_prediction/pipeline/`):
- Data ingestion: stage_01_data_ingestion
- Prepare base model: stage_02_prepare_base_model
- Training: stage_03_training
- Evaluation: stage_04_evaluation

Use `main.py` to run the whole pipeline in sequence.

## Output / Artifacts
- artifacts/data_ingestion/         -> downloaded zip and unzipped images
- artifacts/prepare_base_model/     -> base_model.h5, base_model_updated.h5
- artifacts/prepare_callbacks/      -> checkpoint_dir, tensorboard_log_dir
- artifacts/training/               -> trained model, training logs
- scores.json                       -> saved evaluation metrics

## Troubleshooting (common issues)
- Logging: ensure `logging.basicConfig(level=...)` uses lowercase `level`.
- Config not found: run python from project root or ensure CONFIG_FILE_PATH resolves correctly.
- Loss/labels mismatch: use `categorical_crossentropy` for one-hot labels or `sparse_categorical_crossentropy` with `class_mode="sparse"` in generators.
- Loading saved models: load with `compile=False` and recompile a fresh optimizer before calling `.fit()` to avoid optimizer/variable mismatch.
- Eager execution errors: avoid calling `.numpy()` inside graph-backed functions; for debugging enable `tf.config.run_functions_eagerly(True)`.

## Development notes
- Keep dataclass field names in `entity/config_entity.py` consistent with how `ConfigurationManager` constructs them.
- Update `config/*.yaml` when adding new pipeline options.
- Use the `artifacts/` folder for reproducible outputs; keep it in .gitignore if large files exist.

