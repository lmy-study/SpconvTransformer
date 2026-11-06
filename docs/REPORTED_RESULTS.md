# Ablation studies

We compare sparse convolution, Transformer, sum fusion and concatenation fusion on KITTI validation at IoU 0.7. Values follow the tables in our paper. Each method lists 3D easy/moderate/hard/mean followed by BEV easy/moderate/hard/mean.

```text
Average Precision AP_R11 (%) for 3D and Bird's-Eye View (BEV) Object Detection.
Method
3D Detection (IoU = 0.7)
BEV Detection (IoU = 0.7)

Easy
Mod.
Hard
mAP
Easy
Mod.
Hard
mAP
Spconv
87.58
77.90
76.36
80.61
89.65
87.26
85.12
87.34
Transformer
87.16
77.17
75.29
79.87
89.63
86.94
84.52
87.03
SpconvTranformer+Sum
88.28
77.95
76.86
81.03
89.97
87.44
86.25
87.89
SpconvTranformer+ Concat
88.50
78.33
77.53
81.45
90.07
87.58
86.63
88.09
IMP(Spconv)
+0.92
+0.43
+1.17
+0.84
+0.42
+0.32
+1.51
+0.75
IMP(Transformer)
+1.34
+1.16
+2.24
+1.58
+0.44
+0.64
+2.11
+1.06
Average Precision AP_R40 (%) for 3D and Bird's-Eye View (BEV) Object Detection.
Method
3D Detection (IoU = 0.7)
BEV Detection (IoU = 0.7)

Easy
Mod.
Hard
mAP
Easy
Mod.
Hard
mAP
Spconv
89.75
80.81
77.83
82.80
92.39
88.28
87.08
89.25
Transformer
88.20
78.55
75.69
80.81
92.30
88.08
85.47
88.62
SpconvTranformer+Sum
90.84
80.98
78.50
83.44
94.64
88.65
87.93
90.41
SpconvTranformer+ Concat
90.65
81.59
79.20
83.81
92.81
88.61
88.11
89.84
IMP(Spconv)
+0.90
+0.78
+1.37
+1.01
+0.42
+0.33
+1.03
+0.59
IMP(Transformer)
+2.45
+3.04
+3.51
+3.00
+0.51
+0.53
+2.64
+1.22


```
