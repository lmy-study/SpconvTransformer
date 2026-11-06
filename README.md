# SpconvTransformer

Code repository for our paper **Dual Architecture Fusion Vehicle Lidar 3D Target Detection** (CAIBDA 2026).

**Muyun Li, Weizheng Jin, Yuan Sun**  
[Paper DOI: 10.1109/CAIBDA70336.2026.11621474](https://doi.org/10.1109/CAIBDA70336.2026.11621474)

We propose SpconvTransformer, a dual-branch backbone for LiDAR-based 3D vehicle detection. Our method combines sparse convolution for local geometry and semantic features with voxel attention for broader contextual information. We concatenate the two BEV feature streams and use an RPN for classification, box regression and orientation prediction.

## Architecture

KITTI points → hard voxelization (maximum five points) → mean XYZ/intensity VFE → two branches:

- Sparse CNN: OpenPCDet's SECOND `VoxelBackBone8x`, with submanifold convolutions and three stride-two sparse stages, then vertical compression; output `(B,128,2,200,176)`.
- Transformer: independent feature embedding and three sparse query-generation stages, each followed by one non-residual attention module and two residual submanifold attention modules. Local 3×3×3 offsets plus dilated offsets at strides 2 and 4 are looked up by sorted integer coordinate keys. Attention is multi-head, position-biased and chunked, with batch isolation and no dense N×N matrix; output `(B,64,5,200,176)`.
- Flatten channel and height dimensions independently, concatenate 256 and 320 channels into `(B,576,200,176)`.
- OpenPCDet two-scale `BaseBEVBackbone`: 128/256 channels, five extra convolutions per scale and two 256-channel deconvolutions. Concatenate to 512 channels.
- Car-only `AnchorHeadSingle`: two orientations, producing 2 class logits, 14 box deltas and 4 direction logits per location. Real anchor assignment, focal classification loss, box regression, direction loss, decoding, rotated NMS and KITTI evaluation come from OpenPCDet.

See [implementation details](docs/IMPLEMENTATION.md) for module organization and tensor dimensions.

## Setup

Use Linux with an NVIDIA GPU for full integration. The installation example below uses Python 3.10, PyTorch 2.1.2/CUDA 11.8 and spconv-cu118 2.3.6. Build OpenPCDet's CUDA extensions with a matching local CUDA toolkit. Follow the [upstream installation guide](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md) if compilation requires adjustments.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install 'numpy<2' spconv-cu118==2.3.6
# From this repository root:
git clone https://github.com/open-mmlab/OpenPCDet.git ../OpenPCDet
git -C ../OpenPCDet checkout 233f849829b6ac19afb8af8837a0246890908755
pip install -r ../OpenPCDet/requirements.txt
pip install -e ../OpenPCDet
pip install -e .
python -m unittest discover -s tests
python tests/smoke_cuda.py
```

Upstream is an external Apache-2.0 dependency, not bundled. Keep its license and notices when redistributing upstream source.

## KITTI preparation

Obtain KITTI object detection point clouds, calibration, images and training labels under the dataset terms. Place them in `../OpenPCDet/data/kitti/`:

```text
kitti/
  ImageSets/{train,val,trainval,test}.txt
  training/{velodyne,calib,image_2,label_2}/
  testing/{velodyne,calib,image_2}/
```

Use upstream KITTI `ImageSets` (3,712 training / 3,769 validation examples). Images are used for image dimensions and KITTI projection/evaluation; this detector consumes LiDAR features only. From the OpenPCDet root, generate infos and the augmentation database:

```bash
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

The model config inherits upstream KITTI defaults: range `[0,-40,-3,70.4,40,1]`, voxel size `[0.05,0.05,0.1]`, five points per voxel. Training shuffles points before voxelization. Augmentation settings are defined in the inherited KITTI dataset configuration.

## Training and evaluation

Run from this repository root after setup and KITTI preprocessing:

```bash
python tools/run.py train --openpcdet ../OpenPCDet --cfg_file "$PWD/configs/spconvtransformer.yaml" --batch_size 1 --epochs 100
```

The wrapper registers the backbone and fusion module, changes into upstream `tools`, and executes the actual upstream entry point. Training uses Adam, 100 epochs and batch size one. The learning rate and schedule are specified in the configuration.

Evaluate a trained checkpoint with the following command. Checkpoints are not included in this repository:

```bash
python tools/run.py test --openpcdet ../OpenPCDet --cfg_file "$PWD/configs/spconvtransformer.yaml" --ckpt /absolute/path/to/checkpoint_epoch_100.pth
```

Read both AP_R11 and AP_R40 outputs from the KITTI evaluator; do not substitute one for the other. For comparison use Car IoU 0.7 and report easy/moderate/hard separately. The paper's aggregate is the mean across these difficulties. AOS is not raw heading accuracy.

## Results

Our experiments on KITTI validation use Car IoU 0.7:

| Metric | Easy | Moderate | Hard | Mean |
|---|---:|---:|---:|---:|
| 3D AP_R11 | 88.50 | 78.33 | 77.53 | 81.45 |
| BEV AP_R11 | 90.07 | 87.58 | 86.63 | 88.09 |
| AOS | 95.30 | 89.36 | 88.65 | 91.10 |

See [ablation results](docs/REPORTED_RESULTS.md) for our comparison of sparse convolution, Transformer, sum fusion and concatenation fusion under AP_R11 and AP_R40.

## Experimental setup

We train our model on the KITTI training split and evaluate it on the validation split.

| Setting | Value |
|---|---|
| Training samples | 3,712 |
| Validation samples | 3,769 |
| Training GPU | Single NVIDIA GeForce GTX 1080 Ti |
| Optimizer | Adam |
| Epochs | 100 |
| Batch size | 1 |
| Training iterations | 371,200 |
| Training time | Approximately 65 hours |
| Detection speed | 7 FPS |
| Evaluation | Car, IoU 0.7; AP_R11, AP_R40 and AOS |

Training time and detection speed are the measurements presented in our paper. The software installation recipe above describes how to set up this repository; it is separate from the experimental hardware and timing information.

## Tests

```bash
python -m unittest discover -s tests
python tests/smoke_cuda.py
```

The unit tests cover attention gradients, empty inputs, batch isolation, coordinate permutation and query chunking. The CUDA smoke test exercises the dual backbone, BEV fusion and backward propagation. Attention uses chunked queries to control temporary gather sizes; memory requirements also depend on active voxel counts and retained training activations.

## Citation

```bibtex
@inproceedings{li2026dual,
  author = {Muyun Li and Weizheng Jin and Yuan Sun},
  title = {Dual Architecture Fusion Vehicle Lidar 3D Target Detection},
  booktitle = {2026 6th International Conference on Artificial Intelligence, Big Data and Algorithms (CAIBDA)},
  year = {2026},
  pages = {1--5},
  doi = {10.1109/CAIBDA70336.2026.11621474}
}
```

The paper PDF, KITTI data and third-party source are not redistributed. No open-source license is assigned to the newly written files pending the author's licensing choice; public availability alone does not grant reuse rights.
