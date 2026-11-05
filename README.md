# PhysicalAI26 HW1

This repository contains the HW1 capture simulator and a reconstruction and
experiment workflow. Python dependencies are managed by [Pixi](https://pixi.sh/)
using the checked-in `pixi.lock` file.

## Setup

1. Install Pixi using the instructions at [pixi.sh](https://pixi.sh/).
2. From this directory, create the `habitat` environment:

   ```bash
   pixi install -e habitat
   ```

   The environment provides Python 3.9, Habitat-Sim 0.3.3 with Bullet,
   Habitat-Lab, and the project dependencies. The first install can take a few
   minutes.
3. Check that Habitat-Sim and Habitat-Lab import correctly:

   ```bash
   pixi run -e habitat smoke
   ```

4. Download and extract the Replica apartment scene used by the default
   configuration:

   ```bash
   pixi run -e habitat fetch-replica
   ```

   This creates `replica_v1/apartment_0/`. Keep the scene at that path for the
   default `configs/second_floor.yaml` configuration.

## Capture data

Start the interactive Habitat-Sim collector:

```bash
pixi run -e habitat python load.py
```

The default configuration is `configs/second_floor.yaml`. Captures are written
under `eval/_data/second_floor/` and include RGB and depth PNGs, camera
intrinsics, and a ground-truth pose trajectory. Use `--config` to select a
different configuration or `--output-root` to change the output directory. See
[`docs/load.md`](docs/load.md) for controls, replay mode, and capture details.

## Reconstruction and experiments

Run commands from this repository directory with `pixi run -e habitat`. The
experiment workflow starts by declaring an experiment over a capture, then
reconstructing it:

```bash
pixi run -e habitat python api.py declare \
  --name <experiment-name> \
  --data-dir <capture-directory> \
  --floor 2

pixi run -e habitat python reconstruct.py \
  --experiment experiments/<experiment-name> \
  --no-vis

pixi run -e habitat python api.py explore experiments/<experiment-name>
```

`declare` creates an experiment directory, the first reconstruction measures
pipeline factors and records a baseline, and `explore` displays the results and
writes interactive plots. For the full workflow, SPARQL frame selection, and
query examples, see [`docs/e2e-workflow.md`](docs/e2e-workflow.md).

## More documentation

- [`docs/pixi.md`](docs/pixi.md): environment details and Pixi commands
- [`docs/load.md`](docs/load.md): capture settings, controls, and outputs
- [`docs/reconstruct.md`](docs/reconstruct.md): reconstruction options
- [`docs/api.md`](docs/api.md): experiment and inspection commands
- [`docs/e2e-workflow.md`](docs/e2e-workflow.md): end-to-end walkthrough

## Implementation summary

All TODO blocks in `utils.py` are implemented:
- `depth_image_to_point_cloud`: from-scratch pinhole back-projection using the capture's `intrinsics.json` (no Open3D projection functions)
- `preprocess_point_cloud`: voxel downsampling, normal estimation, FPFH features
- `local_icp_algorithm`: Open3D point-to-plane ICP
- `reconstruct`: frame-to-frame RANSAC -> ICP, pose composition, scene accumulation
- `visualize_and_evaluate`: Mean L2 distance, estimated trajectory in red, GT in black, ceiling removal

## Bonus track: custom ICP

`my_local_icp_algorithm` in `utils.py` is a from-scratch point-to-plane ICP
(cKDTree nearest neighbours, distance threshold 1.5 x voxel size, linearised
point-to-plane update solved by least squares, Rodrigues rotation rebuild,
max 60 iterations, convergence tolerance 1e-6). It is selected with: <flag/option>.
The default pipeline still uses the Open3D implementation.

## Reproducing the results

Final settings: voxel size 0.05, RANSAC seed default, selection query <file or "none">.
(On macOS I ran with `OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=True`.)

Phase 1:
```bash
# 1. Declare the workspace
pixi run -e habitat python api.py declare --name p1_baseline --data-dir eval/first_floor --floor 1

# 2. Seal the baseline (full-batch run)
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=True pixi run -e habitat python reconstruct.py --experiment experiments/p1_baseline --voxel-size 0.05 --no-vis

# 3. Apply the SPARQL filter to drop cheating frames
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=True pixi run -e habitat python reconstruct.py --experiment experiments/p1_baseline --voxel-size 0.05 --selection-query experiments/p1_baseline/queries/frames.rq --no-vis
```
Phase 2: 
```bash
# 1. Declare the workspace
pixi run -e habitat python api.py declare --name p2_take2 --data-dir eval/_data/second_floor/315551006 --floor 2

# 2. Seal the baseline (full-batch run)
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=True pixi run -e habitat python reconstruct.py --experiment experiments/p2_take2 --voxel-size 0.05 --no-vis
```
