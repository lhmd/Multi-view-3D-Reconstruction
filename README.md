# Multi-view 3D Reconstruction

## 运行环境

### 环境配置

```bash
python=3.10.14
pip:
	numpy==1.26.4
	open3d==0.18.0
	opencv-python==4.10.0.82
	torch==2.3.1
	trimesh==4.4.1
```

### 数据集

支持ScanNet与SUNRGBD。

#### ScanNet

示例格式如下：

```bash
.
├── color
│   ├── 0.jpg
│   ├── ......
├── depth
│   ├── 0.png
│   ├── ......
├── intrinsic
│   ├── extrinsic_color.txt
│   ├── extrinsic_depth.txt
│   ├── intrinsic_color.txt
│   └── intrinsic_depth.txt
└── pose
    ├── 0.txt
    ├── ......
```

#### SUNRGBD

```bash
.
├── 0000001-000000020431
│   ├── annotation
│   │   └── index.json
│   ├── annotation2D3D
│   │   ├── index.json
│   │   ├── index.json_20150712014949
│   │   ├── index.json_20150712014954
│   │   ├── index.json_20150712015009
│   │   ├── index.json_20150712015114
│   │   └── index.json_20150712015143
│   ├── annotation2Dfinal
│   │   └── index.json
│   ├── annotation3D
│   │   └── index.json
│   ├── annotation3Dfinal
│   │   └── index.json
│   ├── annotation3Dlayout
│   │   └── index.json
│   ├── depth
│   │   └── 0000002-000000033369.png
│   ├── depth_bfx
│   │   └── 0000002-000000033369.png
│   ├── extrinsics
│   │   └── 20140922010143.txt
│   ├── fullres
│   │   ├── 0000001-000000020431.jpg
│   │   ├── 0000002-000000033369.png
│   │   └── intrinsics.txt
│   ├── image
│   │   └── 0000001-000000020431.jpg
│   ├── intrinsics.txt
│   ├── scene.txt
│   └── seg.mat
├── ......
```

### 重建

```bash
python main.py
```

具体config配置参考`main.py`文件中设置。

## 文档参考

- 最终报告：[最终报告.pdf](docs\最终报告.pdf) 
- 答辩PPT： [结项答辩.pptx](docs\结项答辩.pptx) 
- 参考文献： [基于多视图深度采样的自然场景三维重建.pdf](docs\基于多视图深度采样的自然场景三维重建.pdf) 
