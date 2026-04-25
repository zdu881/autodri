# PERCLOS、眨眼频率与眨眼持续时长的定义与计算

## 1. PERCLOS 的定义

PERCLOS（Percentage of Eye Closure）通常指在给定时间窗内，眼睛处于闭合状态的时间占比。在驾驶疲劳研究中，最常见、最接近标准化的定义是 `P80`，即：

- 眼睑遮挡瞳孔或虹膜达到 `80%` 及以上的时间占比；
- 统计对象是有效观测时间，而不是全部视频时间；
- 经典人工评分体系通常强调困倦相关的缓慢闭眼，不将极短的快速眨眼等同于疲劳闭眼。

这一口径可追溯到：

- Wierwille, W. W., & Ellsworth, L. A. (1994). *Evaluation of driver drowsiness by trained raters*. *Accident Analysis & Prevention*, 26(5), 571-581.
- Dinges, D. F., Mallis, M., Maislin, G., & Powell, J. W. (1998). *Evaluation of techniques for ocular measurement as an index of fatigue and the basis for alertness management*. FHWA-MC-98-006.

## 2. PERCLOS 的计算公式

设：

- \(W\) 为统计时间窗；
- \(c(t)\in[0,1]\) 为时刻 \(t\) 的闭眼程度，`0` 表示全开，`1` 表示全闭；
- `valid(t)` 表示该时刻是否可用；
- \(\Delta t\) 为采样时间间隔。

则经典 `P80` 定义可写为：

\[
\mathrm{PERCLOS}_{P80}(W)=\frac{\sum_{t\in W}\mathbf{1}[c(t)\ge 0.8]\Delta t}{\sum_{t\in W}\mathbf{1}[\mathrm{valid}(t)]\Delta t}\times100\%
\]

若视频以固定帧率采样，上式可写成帧计数形式：

\[
\mathrm{PERCLOS}_{P80}(W)=\frac{N_{\ge 80\%\ closed}}{N_{valid}}\times100\%
\]

其中：

- \(N_{\ge 80\%\ closed}\) 为时间窗内满足“闭合程度大于等于 `80%`”的有效帧数；
- \(N_{valid}\) 为可用于计算的总有效帧数。

## 3. 用 EAR 估计 PERCLOS 的工程化写法

在自动化视觉算法中，PERCLOS 常借助 EAR（Eye Aspect Ratio）近似估计。EAR 的经典来源是：

- Soukupova, T., & Cech, J. (2016). *Real-time eye blink detection using facial landmarks*. In *21st Computer Vision Winter Workshop*.

EAR 公式为：

\[
EAR=\frac{\lVert p_2-p_6\rVert+\lVert p_3-p_5\rVert}{2\lVert p_1-p_4\rVert}
\]

其中 \(p_1,\dots,p_6\) 为单眼的 6 个关键点坐标。眼睛越闭合，EAR 越小。

若对个体先做标定，得到：

- \(EAR_{open}\)：睁眼时的平均 EAR；
- \(EAR_{closed}\)：闭眼时的平均 EAR。

则可构造闭眼程度估计：

\[
\hat c(t)=\frac{EAR_{open}-EAR(t)}{EAR_{open}-EAR_{closed}}
\]

并将其裁剪到区间 \([0,1]\)：

\[
\hat c(t)\leftarrow \min(1,\max(0,\hat c(t)))
\]

于是 `80%` 闭合对应的阈值为：

\[
EAR_{80}=EAR_{open}-0.8\cdot(EAR_{open}-EAR_{closed})
\]

满足

\[
EAR(t)\le EAR_{80}
\]

即可判定该帧达到 `P80` 闭合标准。对应的工程化 PERCLOS 估计为：

\[
\widehat{\mathrm{PERCLOS}}_{P80}(W)=\frac{\sum_{t\in W}\mathbf{1}[\hat c(t)\ge 0.8]\Delta t}{\sum_{t\in W}\mathbf{1}[\mathrm{valid}(t)]\Delta t}\times100\%
\]

或其帧形式：

\[
\widehat{\mathrm{PERCLOS}}_{P80}(W)=\frac{N_{EAR\le EAR_{80}}}{N_{valid}}\times100\%
\]

需要说明的是：`EAR -> 闭眼程度 -> P80` 的映射属于工程实现，不是 Wierwille 或 Dinges 原始人工评分定义本身。论文中建议明确写成“基于 EAR 的 PERCLOS 近似估计”。

## 4. 眨眼频率与眨眼持续时长

### 4.1 眨眼事件定义

一次 blink 通常定义为一个完整的：

`睁眼 -> 闭眼 -> 再次睁眼`

过程。自动化检测时，通常根据 EAR 阈值或闭眼程度阈值识别一次事件的起点和终点。

相关经典文献包括：

- Stern, J. A., Boyer, D., & Schroeder, D. (1994). *Blink rate: A possible measure of fatigue*. *Human Factors*, 36(2), 285-297.
- Caffier, P. P., Erdmann, U., & Ullsperger, P. (2003). *Experimental evaluation of eye-blink parameters as a drowsiness measure*. *European Journal of Applied Physiology*, 89(3-4), 319-325.
- Schleicher, R., Galley, N., Briest, S., & Galley, L. (2008). *Blinks and saccades as indicators of fatigue in sleepiness warnings*. *Ergonomics*, 51(7), 982-1010.

### 4.2 眨眼频率（Blink Rate）

设：

- \(N_{blink}\) 为时间窗内检测到的眨眼次数；
- \(T_{valid}\) 为有效观测时长，单位为秒。

则眨眼频率可表示为：

\[
BR=\frac{N_{blink}}{T_{valid}}
\]

若以 `blinks/min` 表示，则：

\[
BR_{min}=60\cdot\frac{N_{blink}}{T_{valid}}
\]

### 4.3 单次眨眼持续时长（Blink Duration）

设第 \(i\) 次眨眼的开始时刻为 \(t_{start,i}\)，结束时刻为 \(t_{end,i}\)，则该次眨眼持续时长定义为：

\[
D_i=t_{end,i}-t_{start,i}
\]

若按连续闭眼帧数计算，设第 \(i\) 次眨眼包含 \(n_i\) 帧，视频帧率为 `fps`，则：

\[
D_i=\frac{n_i}{fps}
\]

平均眨眼持续时长为：

\[
\bar D=\frac{1}{N_{blink}}\sum_{i=1}^{N_{blink}}D_i
\]

若需要报告总体闭眼时间，也可定义：

\[
T_{blink,total}=\sum_{i=1}^{N_{blink}}D_i
\]

## 5. 疲劳研究中的解释建议

从上述文献可归纳出以下较稳妥的解释：

- `PERCLOS` 主要反映较长时间的眼睑下垂和持续闭眼，对疲劳和困倦较敏感；
- `Blink Rate` 可能随疲劳增加而变化，但也容易受到任务负荷、视觉任务类型、对话行为等因素影响；
- `Blink Duration`、长闭眼比例、睁眼恢复时间等参数，通常比单独的眨眼次数更稳定地反映疲劳水平；
- 在自然驾驶场景中，常将 `PERCLOS + Blink Duration + Blink Rate + 头姿/视线信息` 联合使用，而不是只依赖单一指标。

对应参考来源：

- Bergasa, L. M., Nuevo, J., Sotelo, M. A., Barea, R., & Lopez, M. E. (2006). *Real-time system for monitoring driver vigilance*. *IEEE Transactions on Intelligent Transportation Systems*, 7(1), 63-77.
- Sommer, D., & Golz, M. (2010). *Evaluation of PERCLOS based current fatigue monitoring technologies*. In *Proceedings of the 32nd Annual International Conference of the IEEE EMBS*.

## 6. 论文方法部分推荐写法

如果需要在论文中报告指标，建议至少明确以下内容：

1. `PERCLOS` 的定义是否采用 `P80`；
2. 统计窗口长度，例如 `30 s`、`60 s` 或滑动窗口；
3. 分母是否仅包含有效帧；
4. 是否排除极短的快速眨眼；
5. `Blink Rate` 的单位是否为 `blinks/min`；
6. `Blink Duration` 是单次时长、平均时长，还是总闭眼时长；
7. 若使用 EAR，需说明阈值来源是固定阈值还是个体标定阈值。

可直接用于方法部分的简写如下：

> 本研究采用 `PERCLOS-P80` 作为闭眼时间占比指标，定义为在给定时间窗内眼睑闭合程度达到或超过 `80%` 的有效时间占总有效观测时间的比例。对于基于视频的自动检测，首先通过面部关键点计算眼睛纵横比（EAR），再依据个体标定的睁眼与闭眼 EAR 值将 EAR 映射为闭眼程度，并据此估计 `PERCLOS-P80`。此外，眨眼频率定义为单位有效观测时间内的眨眼次数，眨眼持续时长定义为单次 blink 从闭眼开始到重新睁眼结束的时间长度。

## 7. 参考文献与链接

- Wierwille, W. W., & Ellsworth, L. A. (1994). *Evaluation of driver drowsiness by trained raters*. *Accident Analysis & Prevention*, 26(5), 571-581. https://pubmed.ncbi.nlm.nih.gov/7999202/
- Dinges, D. F., Mallis, M., Maislin, G., & Powell, J. W. (1998). *Evaluation of techniques for ocular measurement as an index of fatigue and the basis for alertness management*. FHWA-MC-98-006. https://rosap.ntl.bts.gov/view/dot/2518/dot_2518_DS1.pdf
- Soukupova, T., & Cech, J. (2016). *Real-time eye blink detection using facial landmarks*. https://cmp.felk.cvut.cz/ftp/articles/cech/Soukupova-TR-2016-05.pdf
- Stern, J. A., Boyer, D., & Schroeder, D. (1994). *Blink rate: A possible measure of fatigue*. *Human Factors*, 36(2), 285-297. https://pubmed.ncbi.nlm.nih.gov/8070793/
- Caffier, P. P., Erdmann, U., & Ullsperger, P. (2003). *Experimental evaluation of eye-blink parameters as a drowsiness measure*. *European Journal of Applied Physiology*, 89(3-4), 319-325. https://pubmed.ncbi.nlm.nih.gov/12736840/
- Schleicher, R., Galley, N., Briest, S., & Galley, L. (2008). *Blinks and saccades as indicators of fatigue in sleepiness warnings*. *Ergonomics*, 51(7), 982-1010. https://pubmed.ncbi.nlm.nih.gov/18568959/
- Bergasa, L. M., Nuevo, J., Sotelo, M. A., Barea, R., & Lopez, M. E. (2006). *Real-time system for monitoring driver vigilance*. *IEEE Transactions on Intelligent Transportation Systems*, 7(1), 63-77. https://doi.org/10.1109/TITS.2006.869598
- Sommer, D., & Golz, M. (2010). *Evaluation of PERCLOS based current fatigue monitoring technologies*. https://pubmed.ncbi.nlm.nih.gov/21095770/
