# TsingYun Vision Aiming - Task 1 & Task 2 详细开发与调试日志

## Task 1: ArUco 姿态估计与相机标定 (Camera Calibration & ArUco Pose Estimation)

### 1. 相机标定 (`calibrate.py`)
在计算机视觉中，为了消除透镜引起的画面畸变，并建立真实物理世界 3D 坐标系与 2D 像素坐标系的映射关系，我们需要先求出相机的内参矩阵。
- **角点提取**：利用 `cv2.findChessboardCorners` 在多张不同角度拍摄的黑白棋盘格图像中寻找内角点，并通过 `cv2.cornerSubPix` 进行亚像素级精确化。
- **矩阵解算**：构建对应的 3D 物理空间坐标点序列，最后调用 `cv2.calibrateCamera` 算出相机的内参矩阵（Camera Matrix，包含焦距 `fx, fy` 和光心 `cx, cy`）以及畸变系数（Distortion Coefficients `k1, k2, p1, p2, k3`）。

### 2. ArUco 识别与 3D 渲染 (`aruco_render.py`)
有了相机内参后，我们通过识别已知物理尺寸的 ArUco 码来实现完整的 AR 渲染：
- **目标检测**：利用 OpenCV 的 `cv2.aruco.ArucoDetector` 及对应的预设字典检测视频帧中的 ArUco 标签，获得标签的 ID 以及 4 个 2D 像素角点坐标。
- **位姿解算 (PnP)**：已知 ArUco 码的物理边长，以标签中心为原点构建 3D 世界坐标 `[[-s/2, s/2, 0], [s/2, s/2, 0], ...]`。将 3D 点、提取到的 2D 像素点、以及前一步算出的相机内参矩阵传入 `cv2.solvePnP`，得出该标签相对于相机的 3D 旋转向量（`rvec`）和坐标平移向量（`tvec`）。
- **投影与渲染**：利用 `cv2.projectPoints` 将我们虚构的 3D 坐标轴或 3D 物体模型（如立体方块）从 3D 空间反向投影到当前的 2D 图像上，并利用 `cv2.line` 画出，实现稳定跟随真实物体的 AR 效果。

---

## Task 2: MNIST 装甲板检测与分类 (Detector & Neural Network Classifier)

### 1. 传统视觉检测主流程 (`detector.py`)
- **颜色过滤与多边形拟合 (`detect_bbox`)**：
  将图像转为 numpy 数组并分离 RGB 通道，利用逻辑运算 `((r > threshold) & (r > g + 20) & (r > b + 20))` 获取纯红色的 LED 装甲板外框 Mask。
  使用 `cv2.getStructuringElement` 结合开运算（`cv2.morphologyEx`）消除零星噪点，接着用 `cv2.findContours` 提取轮廓。对于面积大于 100 的轮廓，使用 `cv2.approxPolyDP` 进行多边形拟合，最终保留正好有 4 个顶点的四边形作为候选装甲板。
- **角点排序 (`order_corners`)**：
  为保证 PnP 解算及图像透视变换的一致性，通过坐标排序将角点严格定义为 `左上(LU), 右上(RU), 右下(RD), 左下(LD)`。先按 X 轴分出左右两拨，再在每一拨里按 Y 轴分出上下，确保顺序万无一失。
- **仿射透视变换 (`crop_bbox`)**：
  得到候选的 4 个角点后，利用 `cv2.getPerspectiveTransform` 和 `cv2.warpPerspective` 进行透视矫正，将倾斜的装甲板抠出并拉伸成 128x128 的正视角图像，四周预留 `15%` 的 Margin。
- **PnP 位姿解算 (`solve_pnp`)**：
  依照装甲板的实际物理宽高，构建其处于原点的 3D 坐标点阵列，与 `detect_mnist_board` 中对应的像素坐标做一一映射，调用 `cv2.solvePnP(flags=cv2.SOLVEPNP_ITERATIVE)` 完成从图像追踪到 3D 空间测算的任务，并将算出的位姿存入 `Detection` 数据类中。

### 2. 深度学习模型构建与训练 (`train.py` & `model.py`)
- **网络架构 (`MNISTClassifier`)**：搭建了一个最基础的多层感知机（MLP）：`Flatten()` 将 28x28 展平为 784，随后经过 `Linear(784, 128) -> ReLU() -> Linear(128, 10)` 算出 10 个数字分类的 Logits。
- **模型训练**：使用 PyTorch `DataLoader` 读取 MNIST 数据集，配合 `Adam` 优化器与 `CrossEntropyLoss` 进行反向传播训练，并在验证集上查看准确率（5 个 Epoch 后可达 97% 以上），最后通过 `torch.save` 将权重保存至本地的 `models/mnist_classifier.npz`。
- **图像预处理流水线 (`preprocess_mnist_crop`)**：将 128x128 的正视角彩色图转化为灰度，压缩为 28x28，然后执行 `x = (x - 0.1307) / 0.3081` 这个极其重要的归一化操作，把输入分布对齐至训练模型时见过的分布，并拼装为 `(1, 1, 28, 28)` 的张量。
- **前向推理 (`predict_mnist_digit`)**：将张量送入被置为 `eval()` 模式的模型中，在 `with torch.no_grad():` 上下文中进行推理（省显存、提速）。最后利用 `F.softmax` 计算出各类别的概率值，通过 `torch.argmax` 抽出概率最高项作为分类结果。

---

## 🐞 重点调试记录与踩坑复盘 (Debugging Details)

在前后端联调测试期间（跑 `pytest` 和本地 Debug 脚本），我们遭遇并攻克了 4 个典型 BUG：

### BUG 1：测试脚手架注入的数据类型不匹配
- **报错现场**：运行 `uv run pytest tests/python/test_task2_mnist_model.py` 报错 `AttributeError: 'list' object has no attribute 'shape'`。
- **根因分析**：在 `test_model_classifier_default_raises_not_implemented` 这个用例中，测试框架强行向我们的 `classify_mnist_digit` 传入了一个 Python 原生三维列表 `[[(255, 255, 255)]]` 用以检查桩函数，而非标准的 OpenCV 图像（`np.ndarray`）。由于我们的代码没有做类型检查，直接读取 `.shape` 导致崩溃。
- **修复方案**：在 `preprocess_mnist_crop` 开头加入阻断防御，将列表强转：
  ```python
  if not isinstance(board_crop, np.ndarray):
      board_crop = np.array(board_crop, dtype=np.uint8)
  ```

### BUG 2：模型极度眼盲，全部预测为数字 1
- **报错现场**：自写脚本读取游戏仿真环境的截屏并抠图后，送进模型预测的结果极其诡异：数字 3 和 7 都被预测为 1，且最高置信度甚至不超过 0.6。
- **根因分析**：编写 Debug 脚本将进入模型前一刻的 Tensor 值打印出来，发现张量四周的值是 `-0.06`（深色背景），但中间赫然有一圈高达 `0.36` 甚至更高的亮色四方环！
  原生 MNIST 模型是在**纯黑背景+纯白数字**的数据集下训练出的，根本没见过边框。而仿真截图里的装甲板有刺眼的红色外框，经过灰度转换后这个红框的灰度值达到了约 100，在归一化后成了画面里一条极其醒目的“干扰特征”。这让分类器完全懵逼。
- **修复方案**：引入了经典的图像二值化手段解决“域偏移（Domain Shift）”问题。在 `cv2.resize` 前，增加 `_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)`。由于红框灰度在 100 左右，背景更低，因此 150 的阈值一刀切去除了所有无关干扰，仅保留 255 的纯白数字，张量分布瞬间与训练集完全对齐。

### BUG 3：高傲的置信度阈值导致目标全丢
- **报错现场**：虽然模型修复了眼盲症，但跑 `test_task2_detector.py` 还是全部挂掉，报错提示 `AssertionError: detector produced no detections`（找不着任何目标）。
- **根因分析**：继续剖析 Tensor。通过分析 `crop_bbox` 的透视逻辑得知，它把外围红框按 15% 的 Margin 缩进了 128x128 画布。算上游戏本身的留白，当这幅图被压扁到 28x28 时，实际纯白色数字在中间占据的像素只有大概 10x12 个。
  这可比 MNIST 官方数据集里的 20x20 个像素小了一倍！这种微小但致命的数据特征偏移（Feature Scaling Shift）导致模型虽然能认出它是 3 或者 7，但概率（Confidence）只有可怜的 0.5 到 0.7 之间。而我之前硬编码写了 `if confidence > 0.8:` 进行过滤，导致稍微不确定的结果惨遭全量过滤。
- **修复方案**：在 `detect_mnist_board` 中将信度过滤阈值从苛刻的 `0.8` 放宽至合理的 `0.4`。改完后所有测试项一路绿灯通过。

### BUG 4：IDE 的静态类型检查报警
- **报错现场**：VSCode (Pyright) 总是针对 `logits = model(input_tensor)` 这一行画红色波浪线，警告 `Expected a callable, got object`。
- **根因分析**：为了对初学者隐藏繁杂的 PyTorch 特性，课程框架组给 `predict_mnist_digit(model: object, ...)` 的类型注解塞了一个最基础的 `object` 类。Python 语法认为基础的 `object` 实例没有实现 `__call__` 魔术方法，自然不能像函数一样被加上括号调用。
- **修复方案**：修正了类型提示。将 `model.py` 中涉及到该参数和返回值的地方由 `object` 修正为更为精准的 `torch.nn.Module`，帮助 IDE 理解它是个可调用的神经网络模块。波浪线当即消失。
