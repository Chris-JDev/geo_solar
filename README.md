# pvx-genai

Working MVP: 2-point georeferencing (pixel ↔ WGS84) and boundary polygon export to GeoJSON.

## Setup
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run
```bash
streamlit run app/app.py
```

## Create a GeoTIFF from a bounding box (leafmap)
In the app, use **Create GeoTIFF from Bounding Box (leafmap)** to download map tiles for a WGS84 bbox and write a georeferenced GeoTIFF (e.g. `data/site.tif`).

Then upload the `.tif` back into the app and choose **Georeference mode → Auto-detect (recommended)** to read GeoTIFF CRS/transform automatically.

## Synthetic data generation (for cVAE training later)
Generate a dataset of conditioning/target mask pairs:
```bash
python -m train.synth_generator --n_samples 5000 --preview
```

This produces folders like:
- `data/synthetic/sample_000001/cond.npy` (float32, shape `(2, size, size)`, values in `{0,1}`)
	- `cond[0]`: boundary mask
	- `cond[1]`: keepout mask
- `data/synthetic/sample_000001/target.npy` (float32, shape `(1, size, size)`, values in `{0,1}`)
	- baseline row stripes clipped to boundary and avoiding keepouts
- `data/synthetic/sample_000001/meta.json` (pitch, keepout counts, etc.)

If `--preview` is set, it also writes `data/synthetic/preview.png` (a 3×3 grid showing boundary/keepouts/target).

## Train the cVAE
After generating synthetic data, train a small conditional VAE:
```bash
python -m train.train_cvae --data_dir data/synthetic --epochs 10 --batch_size 16 --lr 1e-3
```

Outputs:
- `models/layout_cvae.pt` (PyTorch checkpoint containing model weights + config)
- `data/runs/epoch_01.png`, `data/runs/epoch_02.png`, ... (4 sampled masks per epoch)
