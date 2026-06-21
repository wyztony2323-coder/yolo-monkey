# Gibbon-YOLOv8: 海南长臂猿检测实验工程

本项目面向海南长臂猿在雨林环境中的图像与视频检测，基于 Ultralytics YOLOv8s 进行结构改造、损失实验和真实视频诊断。项目重点不是简单堆叠模块，而是围绕真实场景中暴露出的三个问题进行系统验证：

- 小目标定位质量不足：mAP50 接近，但 mAP50-95 提升困难。
- 弱可辨小目标召回：雾天、逆光、树冠遮挡下不能牺牲 P2 小目标检测能力。
- 人类/黑影误检：宣传片中人类黑发、深色衣服和背景黑块容易被误识别为长臂猿。

## 当前最终结论

当前最稳的单帧网络结构为：

```text
YOLOv8s Backbone
+ YOLOv8s PAN-FPN Neck
+ P2/P3/P4/P5 四尺度 Detect Head
+ ResPAM@P2
= yolov8s_p2_respam_p2only
```

真实视频实验表明：

- **P2 检测头必须保留**。P2 降级为辅助注入后，验证集指标看似正常，但真实视频 #7 召回会断崖式下降。
- **不应强行改写 P2 分类/语义路径**。P2-DecoupledHead 会破坏弱可辨小目标响应，并在极端逆光视频中引入假阳性。
- **最终视频部署优先 TrackRecover**。它不改 YOLO 网络，只在视频层恢复与稳定轨迹匹配的低置信候选，对 #7 提升明显且 #11 保持安全。

推荐部署链路：

```text
respam_p2only 单帧检测
-> low-confidence candidate pool
-> TrackRecover temporal recovery
-> final video detections
```

## 核心文件

| 文件 / 目录 | 作用 |
|---|---|
| `train_k_paper_clean_3datasets.py` | 主训练入口，包含各模型类型、训练配置、测试集评估与实验记录 |
| `ultralytics_rainforest.py` | 自定义模块与 Ultralytics patch，包括 ResPAM、AuxDetect、P2SemanticClsDetect、GibbonQualityDetectionLoss |
| `cfg/` | YOLOv8s-P2 系列 YAML 结构配置 |
| `infer_videos_diag.py` | 视频推理与诊断，可输出标注视频、逐帧统计、P2/P3 特征热力图 |
| `diag_clip.py` | 指定时间片段的细粒度视频诊断 |
| `val_conf_sweep.py` | 验证集置信阈值扫描 |
| `tools/infer_videos_track_recover.py` | TrackRecover 视频级低置信候选恢复 |
| `tools/infer_videos_temporal_p2.py` | P2 feature EMA temporal memory 实验 |
| `hnq_refiner.py` | crop-based HNQ refiner 模型与工具函数 |
| `tools/build_hnq_dataset.py` | 构建 HNQ crop 候选数据集 |
| `tools/train_hnq_refiner.py` | 训练 HNQ refiner |
| `tools/infer_hnq.py` | HNQ 推理、HNQ+TrackRecover 评估 |

## 已验证的主要方向

### 1. 保留 P2 四头检测

`yolov8s_p2_respam_p2only` 是当前自定义结构中最稳的单帧网络。P2 直接出框对弱可辨小目标非常关键。

### 2. TrackRecover 视频系统层

TrackRecover 使用双阈值候选池：

```text
high_conf detections: 正常输出并更新轨迹
low_conf candidates: 只有匹配稳定轨迹时才恢复
```

该方法对视频部署最有价值，尤其能恢复雾天/遮挡下连续存在但单帧置信度偏低的目标。

### 3. HNQ crop-based refiner

HNQ 是旁路质量分支：

```text
YOLO candidate crop + geometry vector
-> HNQ refiner
-> p_gibbon / q_box / bbox_delta
```

当前结论：HNQ 不污染主检测路径，安全性较好，但单独收益弱于 TrackRecover。HNQ 暂保留为 hard negative / 质量过滤研究分支，不作为最终主线。

### 4. GibbonQualityLoss-v1

已实现小目标 NWD 辅助定位损失：

```text
L_total = L_yolo + 0.10 * L_small_nwd
```

初步训练结果未超过 `respam_p2only` baseline，暂不作为主线。后续更值得尝试的是基于人工确认框的 hard negative suppression loss。

## 典型命令

训练最终单帧结构：

```powershell
python train_k_paper_clean_3datasets.py `
  --model_type yolov8s_p2_respam_p2only `
  --single_fold `
  --output_dir result_p2_respam_p2only
```

训练 GibbonQualityLoss-v1 对照：

```powershell
python train_k_paper_clean_3datasets.py `
  --model_type yolov8s_p2_respam_p2only_gql_nwd `
  --single_fold `
  --output_dir result_gql_nwd `
  --epochs 300
```

运行 TrackRecover 视频评估：

```powershell
python tools/infer_videos_track_recover.py `
  --weights result_p2_respam_p2only_ft800/fold_1/weights/best.pt `
  --indices "1,2,7,10,11" `
  --out_dir result_track_recover_eval `
  --high_conf 0.25 `
  --low_conf "0.05" `
  --match_iou "0.25" `
  --max_miss "2"
```

构建并训练 HNQ：

```powershell
python tools/build_hnq_dataset.py `
  --weights result_p2_respam_p2only_ft800/fold_1/weights/best.pt `
  --out_dir hnq_dataset `
  --device cpu `
  --hard_video_specs "11:0:-1" `
  --hard_video_background_per_frame 3

python tools/train_hnq_refiner.py `
  --csv hnq_dataset/candidates.csv `
  --out_dir result_hnq_refiner_clean
```

## 不建议提交到 GitHub 的内容

该目录下包含大量训练输出、权重、缓存和诊断产物。推送代码仓库时建议排除：

- `*.pt`
- `result_*/`
- `hnq_dataset/crops/`
- `dataset_split_631/`
- `__pycache__/`
- `*.mp4`
- `*.avi`
- `*.log`
- 大型图片/特征可视化输出

建议只提交源码、配置、文档和少量可复现实验脚本。

## 项目状态

当前推荐的最终方案：

```text
单帧模型：yolov8s_p2_respam_p2only
视频部署：respam_p2only + TrackRecover
候选研究分支：HNQ refiner、Hard Negative Loss
已淘汰方向：P2 auxiliary-only、P2-DecoupledHead、P2 feature EMA、低照度强增强、继续堆叠 P2 模块
```
