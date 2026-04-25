# ONNXRuntime CUDA 回退 CPU 问题诊断与修复方案

## 1. 问题现象

在 `adri` 环境中运行 ONNX 推理时，日志会出现类似报错：

```text
Failed to load library libonnxruntime_providers_cuda.so with error:
libcudnn.so.8: cannot open shared object file: No such file or directory
Failed to create CUDAExecutionProvider
```

随后 `onnxruntime` 会自动回退到：

```text
CPUExecutionProvider
```

这意味着：
- Python 能正常运行
- `onnxruntime` 能正常导入
- GPU 机器本身没有问题
- 但 CUDA provider 在真正初始化时失败了

---

## 2. 已确认的诊断结论

### 2.1 GPU 和驱动本身正常

`nvidia-smi` 正常，机器可见多张 `RTX 4090`，不是硬件问题。

### 2.2 adri 环境里的 onnxruntime 支持 CUDA provider

在 `adri` 环境中检查：

```python
import onnxruntime as ort
print(ort.get_available_providers())
```

结果包含：

```text
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'AzureExecutionProvider', 'CPUExecutionProvider']
```

说明安装的不是纯 CPU 版 ORT。

### 2.3 直接创建 session 时会退回 CPU

测试：

```python
import onnxruntime as ort
s = ort.InferenceSession(
    "models/scrfd_person_2.5g.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
print(s.get_providers())
```

在当前环境下结果是：

```text
['CPUExecutionProvider']
```

### 2.4 根因不是“没装 cuDNN”，而是“运行时找不到 cuDNN”

检查 `libonnxruntime_providers_cuda.so` 的依赖：

```bash
ldd /data/home/sim6g/anaconda3/envs/adri/lib/python3.10/site-packages/onnxruntime/capi/libonnxruntime_providers_cuda.so
```

可以看到：

```text
libcudnn.so.8 => not found
```

但在 `adri` 环境中，`libcudnn.so.8` 实际是存在的：

```text
/data/home/sim6g/anaconda3/envs/adri/lib/libcudnn.so.8
```

因此真正的问题是：

- `adri` 环境里有 `libcudnn.so.8`
- 但当前运行方式下，动态链接器没有去 `adri/lib` 里找
- 所以 ORT 的 CUDA provider 初始化失败

### 2.5 关键验证结果

当前默认环境变量：

```bash
echo $LD_LIBRARY_PATH
```

结果为空。

当临时补上：

```bash
LD_LIBRARY_PATH=/data/home/sim6g/anaconda3/envs/adri/lib:${LD_LIBRARY_PATH}
```

再测试 ORT session：

```python
import onnxruntime as ort
s = ort.InferenceSession(
    "models/scrfd_person_2.5g.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
print(s.get_providers())
```

结果变为：

```text
['CUDAExecutionProvider', 'CPUExecutionProvider']
```

这个结果已经足够说明：

**当前回退 CPU 的主因是 `LD_LIBRARY_PATH` 没有包含 `adri/lib`，导致 ORT 找不到 `libcudnn.so.8`。**

---

## 3. 推荐修复方案

### 方案 A：运行前显式注入 `LD_LIBRARY_PATH`

这是当前最直接、最小改动、风险最低的方案。

建议命令模板：

```bash
export LD_LIBRARY_PATH=/data/home/sim6g/anaconda3/envs/adri/lib:${LD_LIBRARY_PATH}
conda activate adri
python your_script.py
```

如果只想作用于单条命令：

```bash
LD_LIBRARY_PATH=/data/home/sim6g/anaconda3/envs/adri/lib:${LD_LIBRARY_PATH} \
/data/home/sim6g/anaconda3/envs/adri/bin/python your_script.py
```

适用场景：
- 临时跑推理
- 不想改代码
- 不想动系统库配置
- 需要快速恢复 GPU 推理

优点：
- 改动最小
- 立即见效
- 不影响系统级 CUDA 安装

缺点：
- 需要每次运行时显式带上
- 如果别人忘了加，问题会再次出现

---

## 4. 可选长期方案

### 方案 B：把 `adri/lib` 写入环境初始化脚本

可以在 `conda activate adri` 后自动带出：

```bash
export LD_LIBRARY_PATH=/data/home/sim6g/anaconda3/envs/adri/lib:${LD_LIBRARY_PATH}
```

例如写入用户 shell 启动脚本，或者写入 conda env 的激活脚本。

优点：
- 使用时更省事
- 不容易忘

缺点：
- 这属于环境层修改，不适合在没评估前直接动
- 如果机器上有多个 CUDA 相关环境，需要注意相互影响

---

## 5. 不推荐优先做的方案

### 方案 C：重新安装 ORT / CUDA / cuDNN

当前证据显示：
- ORT 已经安装
- CUDA provider 已经可见
- cuDNN 也已经存在于 `adri/lib`

所以现在并不是一个“必须重装”的问题。

如果直接重装：
- 风险更高
- 时间更长
- 容易引入新的版本兼容问题

除非后续确认还有别的符号冲突或版本冲突，否则不建议先走这条路。

---

## 6. 建议的执行顺序

推荐按下面顺序处理：

1. 先使用方案 A，验证所有 ONNX 推理脚本都能恢复 CUDA provider
2. 如果稳定，再考虑是否做方案 B，固化到环境初始化
3. 只有在 A/B 都不能稳定工作时，才考虑重装 ORT 或 CUDA/cuDNN

---

## 7. 建议的验证命令

修复后先做最小验证：

```bash
LD_LIBRARY_PATH=/data/home/sim6g/anaconda3/envs/adri/lib:${LD_LIBRARY_PATH} \
/data/home/sim6g/anaconda3/envs/adri/bin/python - <<'PY'
import onnxruntime as ort
s = ort.InferenceSession(
    "models/scrfd_person_2.5g.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
print(s.get_providers())
PY
```

期望输出：

```text
['CUDAExecutionProvider', 'CPUExecutionProvider']
```

然后再跑项目内实际脚本验证，例如：

```bash
LD_LIBRARY_PATH=/data/home/sim6g/anaconda3/envs/adri/lib:${LD_LIBRARY_PATH} \
/data/home/sim6g/anaconda3/envs/adri/bin/python gaze_onnx/experiments/assign_dual_roi.py --help
```

以及真实推理脚本：

```bash
LD_LIBRARY_PATH=/data/home/sim6g/anaconda3/envs/adri/lib:${LD_LIBRARY_PATH} \
/data/home/sim6g/anaconda3/envs/adri/bin/python gaze_onnx/gaze_state_cls.py --help
```

如果后续需要更严格验证，可以在真实推理脚本中打印：

```python
import onnxruntime as ort
print(ort.get_available_providers())
```

并确认 session 实际 provider 里有 `CUDAExecutionProvider`。

---

## 8. 当前结论一句话版

当前 ONNX 回退 CPU 的主因是：

**`adri` 环境里的 `libcudnn.so.8` 已安装，但运行时 `LD_LIBRARY_PATH` 为空，导致 `onnxruntime` 的 CUDA provider 加载不到 cuDNN，于是自动退回 CPU。**
