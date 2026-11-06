# Implementation details

Our detector uses OpenPCDet for KITTI loading, voxelization, box transforms, anchor targets, loss functions, decoding, rotated NMS and evaluation.

## Backbone

The sparse CNN branch uses `VoxelBackBone8x`. The Transformer branch embeds mean voxel features and applies three stride-two sparse query-generation stages. Each stage contains one non-residual attention block and two residual submanifold attention blocks.

Sparse convolution generates query coordinates and initial features. Attention operates on that support using local and dilated offsets. Submanifold blocks preserve the active coordinates. The attention implementation uses four heads, channels 16→32→64→64, offsets at strides 1/2/4, learned relative-position bias, LayerNorm and a GELU feed-forward network. Coordinate lookup includes batch indices to isolate scenes.

Mean VFE consumes XYZ and reflectance. Each voxel retains at most five points.

## BEV and RPN

CNN output: `(B,128,2,200,176)`. Transformer output: `(B,64,5,200,176)`. Flattening height into channels yields 256 and 320 channels; concatenation yields 576.

The implemented RPN uses a 576→128 first scale and a 128→256 second scale. The upsampling branches map 128→256 and 256→256, yielding a 512-channel fused representation. The classification, box and direction outputs have 2, 14 and 4 channels respectively.

## Configuration

`configs/spconvtransformer.yaml` defines the detector and optimizer and inherits KITTI data processing from OpenPCDet. It includes the Car anchor geometry, matching thresholds, loss weights and postprocessing settings. The released configuration selects concatenation fusion. The ablation tables summarize the experiments in our paper; separate sum-fusion and single-branch presets are not included in this release.

## Dependencies

The runtime wrapper registers our modules and executes OpenPCDet's training or evaluation entry point. The installation instructions pin the upstream revision so registry interfaces and inherited configurations remain consistent.
