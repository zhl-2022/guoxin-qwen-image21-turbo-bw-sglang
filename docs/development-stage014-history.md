# Qwen-Image 2.1 Turbo 完整 TP1 开发阶段 014

## 交付状态

完整 TP1 代码已经形成可执行的小包：固定源代码与五组件、严格完整架构检查、真实 native streaming 权重加载、八个 Turbo sigma、256 后继续 512 的同驻留开发采样。010 core、完整 VAE 权重 CPU 探针、011 八步真实接口链、012 完整架构与实际 header 映射的原件均已独立通过。014 的完整 GPU 输出仍待全 17 文件完整校验和实际执行。

这份记录接续 `qwen21-core-development007.md` 与 `qwen21-stage009-execution.md`。所有 005 至 014 的旧候选、小包和原件继续保留。没有改动 native wheel、旧 venv、已有模型权重或赛事平台，没有 push 或赛事提交。

| 证据层 | 截至此文件形成时的状态 |
|---|---|
| 008 原件独立审计 | CRC、11 payload、冻结 bundle 绑定通过；三项成功、两项导入或类定义失败 |
| 009 原件独立审计 | CRC、5 payload、冻结 bundle 绑定通过；两个 selected worker 失败 |
| 009 DiT 真实进展 | 已构造真实模型并进入 forward，缺 native USP 新的 replicated K/V prefix 方法 |
| 009 encoder 真实进展 | 已进入文本 RoPE 构造，native SRT 全局配置未初始化 |
| 010 core | 两个 selected worker 实际 forward 通过；699 个加载框架模块全 SHA 绑定 |
| 14 文件 metadata + VAE 下载 | 实际 VAE 原件中的 14 个完整 SHA 全部绑定源 lock |
| 完整 VAE 权重 CPU 编解码 | 337,740,404 参数、238 keys 严格加载；CPU float32 RGBA 编解码有限 |
| 011 五组件八步接口链 | 真正组件类、pre-norm、drop14、八 sigma、prefix cache、完整 RGBA PNG 通过；随机 tiny 架构 |
| 012 完整架构与实际 header 映射 | 完整 16,219,988,836 参数均在 meta；750/297/238 key/shape 双射通过 |
| 014 完整 GPU runner | 源代码、小包、CPU 数值回归已完成；完整 GPU 执行待前置证据 |
| 正式精度、性能、验收、提交 | 未完成，开发样例不能替代 |

## 009 独立发现与 010 修复

009 原件为 `dist/000-session007-qwen21-core009-raw.zip`，75,086 字节，SHA256 `1fdb4afe28a17d0c3df0be6e34da9217dad9837ddb3dcf3f998fb934e043a3b2`。两个 worker 各自的 1,989 个 native Python 文件前后一致，699 个实际加载的不同框架模块全部匹配固定 native 或 overlay SHA。

DiT 的失败不是导入错误或数值错误。真实 forward 进入 `QwenImage21Attention.attend_sample` 后，旧版 `USPAttention` 缺少 `forward_with_replicated_kv_prefix`。010 在具体模型内继承真实 native USP 类，复制固定上游 SP1 分支：按原顺序连接 prefix/suffix K/V，调用原生 `attn_impl.forward(q,k,v,ctx_metadata)`。该分支有 AST 相等检查；SP 大于 1 明确拒绝，没有移植 NVIDIA JIT 到 HCU。

encoder 失败于真实 native RoPE 的 `_compute_inv_freq`。native SRT 与 MM 各自有一套全局 `ServerArgs`，原探针只初始化了 MM。010 使用真正的 native SRT `ServerArgs`，按其支持的 `model_path="None"` 无 SRT 服务模式创建完整配置对象，再经真正的 setter 发布。实际 Qwen 架构、源代码和权重仍由独立 MM 路径绑定，未合成假的框架或配置类。

| 文件 | 字节数 | SHA256 |
|---|---:|---|
| `dist/session007-qwen21-turbo-core-probe010.zip` | 4,075,073 | `970bf1270fa18d51ab247d9a096c0e2ca6ebb26075cb40b9f3d623592ff0eecb` |
| `scripts/unpack_qwen21_core_probe007_010.py` | 见文件 | `1723c5812e2cd303dfdc588cb646148c3b391c80af2b0ed03c8e1e43b66699d7` |
| 010 `MANIFEST.json` | 见文件 | `73f10aef47106011a31826ce257867cf61ddd3ebd62f94cb23068ab9e275ed0d` |

010 的严格 531 payload 解压、audit-only、16 项源码数学不变量、26 处 native 直接调用签名检查通过。包内另含真实五组件的 tiny CPU 八步接口链 011；它明确是随机小网络，不能替代下面的完整架构及完整权重检查。

后续原件 `dist/000-session007-qwen21-stage010-012-raw.zip` 为 97,912 字节，SHA256 `b6faeea32b4a20eb2cc175df2e477c3d6a47ad5204e974bb4f85bf2344d54dbc`。CRC、25 个 payload 与 009、010 双 bundle 绑定均独立通过。010 DiT 为 478,352 参数、encoder 为 121,056 参数，真实 CPU forward 的有限输出通过；两 worker 分别耗时 34.071、33.668 秒，包含导入与源码核验，不代表完整模型性能。VAE CPU 实际加载及编解码耗时 37.373 秒。011 接口链总耗时 58.705 秒，生成的随机小网络 PNG 完整解码通过，SHA256 `23c30cb776162ead4e7490faa3964db291c962f6ed0a5fb7dab96bbcd5cfdff5`。

上述每个进程均核验 1,989 个 native Python 文件前后一致，均有 CUDA context 初始化；已完成的计算为 CPU 或 meta，未把 context 初始化写成 GPU 推理通过。

## 实际 checkpoint header 与完整架构检查

使用严格 HTTP 206 Range 获取四个固定文件的 safetensors 头部，共 155,976 字节。请求的 tensor payload 为零；Range 被忽略时脚本在读取正文前拒绝。每个头部的 SHA、来源 commit、大小、dtype、shape 和连续 offset 均封存。它们属于真实发布文件的 header 证据，不是完整权重 SHA 证据。

| 组件 | 源 tensor 数 | 由实际 header 推导的参数数 |
|---|---:|---:|
| text encoder | 750 | 8,767,123,696 |
| DiT，两分片 | 297 | 7,115,124,736 |
| RGBA VAE | 238 | 337,740,404 |
| 合计 | 1,285 | 16,219,988,836 |

这些参数均为 BF16。CPU/meta 012 探针用真实 native 类与完整发布者架构构造模型，逐键比较完整 `state_dict` 与上述 header，检查 key 双射、shape、重复与丢失；视觉 qkv 到 qkv_proj 的唯一名称处理保留具体 Qwen 类的原始规则。没有将 tiny 架构的结果代替完整架构。其实际原生结果已独立通过，耗时 62.197 秒。真实 custom-loader 参数数分别为 232、224、0。完整 encoder 仅有两种 meta 位置缓冲：视觉 `inv_freq` 为 float32 `[18]`，native MRotary `cos_sin_cache` 为 BF16 `[262144,128]`，恰由 014 的原始函数重算路径覆盖。

| 文件 | 字节数 | SHA256 |
|---|---:|---|
| `dist/session007-qwen21-full-shapes012.zip` | 33,622 | `3bb8a32ec31f0af64890da5bad45ee8ccc7eb74383317f6114be3734bc6d99a6` |
| `scripts/unpack_qwen21_full_shapes012.py` | 见文件 | `aabe349e617f8b6fb7c9130cc79e49ffb775dc3a1d106d199b12d900be738c08` |

```bash
python3 -B /root/unpack_qwen21_full_shapes012.py --zip /root/session007-qwen21-full-shapes012.zip --target /root/private_data/qwen21-turbo-full-shapes012
/root/private_data/guoxin-wan-env-006-001/venv/bin/python -B /root/private_data/qwen21-turbo-full-shapes012/probe_qwen21_full_shapes012.py --bundle /root/private_data/qwen21-turbo-core-probe010 --expected-bundle-sha256 73f10aef47106011a31826ce257867cf61ddd3ebd62f94cb23068ab9e275ed0d --output /root/private_data/qwen21-full-shapes012-results
```

## 完整权重与严格加载

全 17 文件共 32,451,622,214 字节；四个权重本身共 32,440,133,648 字节。现有 metadata + VAE 阶段为 686,997,222 字节，剩余下载 31,764,624,992 字节。协调端已授权在原生完整架构检查通过后继续，下载仍使用 Qwen 独立 namespace、owner lock、两个 worker、deadline 与完整 SHA。

原始 launch JSON 记录全量下载启动于 UTC `2026-10-24T09:20:14.395908`，PID `29272`、start ticks `286099809`。此次原件仅含进行中的日志，未包含完整 17 文件 verify 成功回执，故本地 gate 保留全量完整性未确认，不能提前标完成。

```bash
python3 -B /root/private_data/qwen21-turbo-stage009-transfer/download_qwen21_turbo009.py download --component all --max-bytes 34000000000 --max-seconds 1800
python3 -B /root/private_data/qwen21-turbo-stage009-transfer/download_qwen21_turbo009.py verify --component all --max-bytes 34000000000 --max-seconds 300
```

严格加载使用真正的 native `load_model_from_streaming_state_dict` 与参数自带 `weight_loader`，不绕过模型的真实 loader。读取由只读 `safe_open` 逐 tensor 提供，不走 native iterator 中可能删除损坏源文件的分支。完整 key/shape 双射先于加载，因此 native legacy bias/norm 缺失回填、未齐 merge buffer 或重复覆盖不能被误认为严格加载成功。

仅两种已知 nonpersistent 位置缓冲允许通过原类或原函数重算；未知 meta buffer 直接失败。native 字节前后均核验。014 还在每个组件加载后、post-load packing 前，抽取不超过 4 个、每个不超过 32 MiB 的首末与 custom-loader 参数，逐项比较 GPU 实值与源 BF16 tensor 的 dtype、shape、`torch.equal` 和原始字节 SHA。

## 013 数值差异与 014 修正

013 尚未执行完整 GPU，已封存为历史候选。复核发现其 decode 把反归一化简写为 `latents * std + mean`。精确上游 Qwen21 pipeline 返回 `std.reciprocal()`，原生及固定 main 的 DecodingStage 均先除该 scaling factor 再加 mean。低精度运算不能把这两种写法视为逐位等价。

实际 Torch 2.10 CPU 回归执行未修改的原始 `scale_and_shift_latents` 与 `_to_unit_range` 函数，014 float32、BF16 均逐元素相同。旧乘法在 512 个 BF16 元素中有 84 个差异，最大绝对差 0.0625。014 同时保持原像素归一化的 BF16 顺序，再转 float32 序列化。证据为 `evidence/session007/qwen21-decode-order014-real-CPU-regression-b.json`。

014 按实际 HIP 显式选用已安装的真正 `RocmPlatform`，为真正 MM args 提供精确 `QwenImage21PipelineConfig` 与八个 sigma，并通过真实 gate 的 `disable()` 选择原始 eager 路径。没有将 CUDA/NVIDIA fast path 伪装为 HCU 支持。pre-final-norm 条件编码通过真实 hook 检查；prefix K/V、latent pack/unpack、sigma timestep 与模型内部 `/1000` 路径保持原始语义。

| 文件 | 字节数 | SHA256 |
|---|---:|---|
| `dist/session007-qwen21-pretrained014.zip` | 60,340 | `8b52a18dd5a86d187fbacd26fcfc75d94b8073b4385557d0d500d6109e5d523a` |
| `scripts/unpack_qwen21_pretrained014.py` | 见文件 | `2ca152c6a8a230c354d5b2f4302d4be0dce39d3d9af8bce4e87355e820890ea9` |

```bash
python3 -B /root/unpack_qwen21_pretrained014.py --zip /root/session007-qwen21-pretrained014.zip --target /root/private_data/qwen21-turbo-pretrained014
/root/private_data/guoxin-wan-env-006-001/venv/bin/python -B /root/private_data/qwen21-turbo-pretrained014/run_qwen21_pretrained014.py --bundle /root/private_data/qwen21-turbo-core-probe010 --expected-bundle-sha256 73f10aef47106011a31826ce257867cf61ddd3ebd62f94cb23068ab9e275ed0d --shape-receipt /root/private_data/qwen21-full-shapes012-results/full-shapes012.json --resolutions 256,512 --output /root/private_data/qwen21-complete-pretrained014-results --max-seconds 1200
```

完整采样先 256，再使用同一已加载模型生成 512；也可明确选择 `256,1024`。每个输出记录 PNG、RGBA 几何、有限张量、八步、实际设备、源与权重绑定、参数实值抽查、内存与包含校验的耗时。它们属于开发样例，不能标为官方质量通过或正式性能。

## 待收原件与状态边界

`scripts/audit_qwen21_stages013.py` 已实际核验完整 VAE、012 meta/header 原件；支持重复 `--bundle` 以分别绑定 VAE 使用的 009 与后续 core，并通过 `--driver` 绑定完整 runner。归档 CRC、payload 清单、SHA、实际 bundle、native 前后、权重完整性、PNG 原始字节和八步证据均独立检查，再安全解压到新目录。

`evidence/session007/qwen21-full014-readiness-gate.json` 已记录 CPU 与完整架构前置全部通过。完整 GPU 的剩余触发条件是：实际全 17 文件 verify 成功、owned transfer 已退出，并通过 runner 自身的 HCU/显存准入检查。该 gate 不代表完整 GPU 已执行。

当 native 导入初始化了 CUDA context，或完整模型已加载到 GPU 但前向中途异常时，不能把缺少完成的图像输出解释为没有 GPU。014 分别记录 context、loaded tensors、compute attempt 与完成的实际 component forward；尚无完成证据时使用未知状态，保留完整异常。正式赛事材料缺口和人工提交边界继续保留。
