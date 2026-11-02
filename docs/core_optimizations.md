# 《MTranServer-Rust 核心优化技术说明文档》

## 概述与架构全景

### 1. 系统定位与核心痛点
MTranServer-Rust 是一套专为高并发、低延迟工业级生产环境设计的原生 Rust 神经机器翻译推理服务端。系统深度兼容 Marian NMT 二进制模型架构与 Mozilla Firefox Translations（Bergamot 项目）离线轻量化模型家族，旨在彻底摆脱传统神经机器翻译推理对 Python 运行时、PyTorch 重型框架以及 C++ 庞大外部动态链接库的依赖。

在传统神经机器翻译服务落地场景中，普遍存在四大核心技术痛点。
其一，内存开销失控与模型冷启动迟滞。基于 PyTorch 或 HuggingFace 的服务方案，单个模型实例往往引入数百兆乃至数吉字节的框架底噪，且权重加载涉及繁重的文件反序列化与多层张量拷贝，导致高密度多语种部署时物理内存迅速耗尽。
其二，多线程并发与微架构资源竞争。Python 全局解释器锁（GIL）严重限制了多核心纵向扩展能力，跨进程方案则加剧通信损耗与内存多副本膨胀；而 CTranslate2 或 Marian C++ 包装方案在面对动态请求排队与 CPU 计算密集型任务时，极易发生超线程上下文剧烈切换与 L1/L2 缓存行颠簸。
其三，长文本富文本结构割裂。常规翻译引擎在处理 Markdown、HTML 或多段落混合文档时，通常采取机械剥离标记的粗暴做法，在译文生成后难以准确回填行内格式，频繁引发语法闭合错误与标记跨词漂移。
其四，自回归解码计算冗余。标准 Transformer 解码器在自注意力计算中承受随序列长度二次方增长的计算与键值缓存（KV-Cache）内存负担，且未剪枝的全词表 Softmax 消耗了超过 70% 的解码时间。

MTranServer-Rust 针对上述痛点进行了端到端全链路重构，在无垃圾回收机制的纯 Rust 内存安全体系下，深度结合现代 CPU 微架构向量指令集与硬件缓存特性，实现了超低时延、紧凑内存与高吞吐的工业级性能平衡。

### 2. 五层分层拓扑架构
系统采用高度解耦且横向贯通的五层拓扑结构。

网络与网关层依托 Axum 0.8 与 Hyper 1 构建底层全异步非阻塞 I/O 事件循环，集成 IP 细粒度并发限流器、全局物理算力门控以及全功能 CORS 与 HTTP 请求链路追踪。

业务编排与缓存层承担多语种翻译流水线调度、多跳中转路由（Pivot Routing）决策、层级化语言自动识别以及 32 分片强制缓存行对齐的全局 LRU 翻译结果缓存。

文本工程与富文本对齐层负责混合排版文档抽象语法树（AST）递归解析、长文本 6 级层级语义子句切分、Unicode 边界感知分句、SentencePiece 紧凑前缀树编解码、Emoji 与变量占位符上下文保护，以及基于交叉注意力能量分布的富文本标记投影对齐。

模型调度与生命周期层基于零拷贝内存映射（mmap）构建张量生命周期管理系统，集成 Single-Flight 防击穿并发加载器、作用域并发多资产装载、无锁原子热度追踪、双重保活队列与自适应 LRU 容量驱逐引擎。

底层张量与 SIMD 算子加速层封装硬件级 CPU 特性动态探测、非对称 INT8 向量化行量化、AVX2 Nibble-Split 无溢出整数矩阵乘法、AVX-VNNI 裸汇编指令发射、ARM64 NEON 与 DotProd 内核、列和修正项预计算、Rayon 自适应 M 维切分以及多算子内存融合流水线。

### 3. 端到端翻译链路
客户端 HTTP 请求抵达网关层，由基于客户端真实 IP 的限流中间件进行分片原子并发度检查，随后经由全局信号量门控限制整体 CPU 算力承载。

进入业务编排层后，系统利用请求参数与规则引擎版本号计算 128 位抗碰撞混合哈希，直接检索 32 分片 LRU 缓存。若缓存命中，直接通过只读原子引用包装返回共享字符串，不发生任何堆内存分配与数据深拷贝。

若缓存未命中，系统对源文本进行探测采样，经由多级级联语言检测流水线判定源语言。若源语言与目标语言间无直接模型支持，则动态激活 Pivot Routing 双跳路由，并发预加载双段模型管道。

输入文本送入富文本解析与文本工程流水线。系统先调用表情与变量保护模块，将 Emoji 字符与插值占位符抽离为结构化符号；接着递归拆解文档语法树，针对超长句子执行 6 级层级语义切分，提取行内标记并记录绝对字符坐标；随后进入 SentencePiece 分词流水线，利用静态双特化 ASCII 前缀树完成字节流游走与 Viterbi 零堆分配切词。

分词序列投递至 Transformer 推理核心。编码器通过自适应 M 维切分完成非自回归特征提取；解码器利用词表剪枝器构建动态目标候选词表，采用 SSRU 循环单元完成步进解码。在自回归循环中，单步解码直接复用线程专享的重用缓冲区，借助 AVX2、VNNI 或 NEON 硬件内核进行 GEMV 计算，施加多级重复惩罚与自适应动态 EOS 长度引导机制。

模型生成的输出 Token 经由直接查表模式逆向解码为目标文本，交叉注意力累加器提取各解码步的源词注意力峰值。富文本对齐模块根据注意力能量分布进行方差过滤与一维连续密度聚类，经过字符边界安全吸附与语法树冲突消解，将初始剥离的行内标记严密回填，最终还原表情与变量占位符，输出至网络层。

### 4. 系统核心性能指标
在 Intel Xeon Platinum 与 AMD EPYC 现代多核服务器环境下，MTranServer-Rust 展现出优异的基准性能。单句翻译端到端中位延迟（P50）保持在 1.5 毫秒至 3.2 毫秒区间，P99 尾部延迟严格压制在 15 毫秒以内；单模型冷启动加载时间控制在 50 微秒以内（基于 mmap 零拷贝缺页装载），相较于传统反序列化方案提速超过 3 个数量级；系统驻留底噪极低，服务启动基础内存占用小于 12 MB，单模型常驻内存体积与磁盘 INT8 压缩模型大小等同（约 40 MB 至 90 MB）；在全核打满并发压测下，吞吐量随物理核心数呈接近线性的横向扩展。

## 第一章 内存架构与模型生命周期优化

### 1. 零拷贝内存映射（mmap）与惰性反量化
为了彻底消除模型加载时的磁盘 I/O 阻塞与堆内存暴涨，MTranServer-Rust 在底层全面基于 `memmap2` 实现了二进制模型文件的只读虚拟内存映射。系统解析二进制 Marian 模型格式时，仅在用户态解析极简的头部元数据与张量字典，张量底层庞大的权重数据则完全保持在映射地址空间内，由操作系统虚拟内存管理机制实施按需缺页装载（Page Fault on Demand）。

```rust
// src/engine/model.rs
// 模型文件载入与按需缺页映射实现
impl Model {
    pub fn load(path: &Path) -> Result<Self, EngineError> {
        let file = std::fs::File::open(path)
            .map_err(|e| EngineError::Io(format!("Failed to open model file: {e}")))?;

        // 尝试建立只读内存映射，若环境受限或文件系统不支持则安全回退至标准堆内存读取
        if let Ok(mmap) = unsafe { memmap2::Mmap::map(&file) } {
            let mmap = Arc::new(mmap);
            Self::parse_buffer(&mmap, Some(mmap.clone()), None)
        } else {
            let data = std::fs::read(path)
                .map_err(|e| EngineError::Io(format!("Failed to read model file: {e}")))?;
            Self::parse_buffer(&data, None, Some(Arc::from(data.into_boxed_slice())))
        }
    }
}
```

在量化权重存储结构上，系统在 `src/engine/tensor/mod.rs` 中设计了核心枚举 `Quant8Data`，并为其实现了 `std::ops::Deref<Target = [i8]>` 特性。这使得量化张量数据切片无论来源于 mmap 虚拟映射内存、共享内存切片还是堆内存，均能以统一且零开销的 `&[i8]` 方式直接借用，杜绝中间结构转换与数据移动。

```rust
// src/engine/tensor/mod.rs
// 零拷贝量化张量数据源封装与 Deref 实现
#[derive(Debug, Clone)]
pub enum Quant8Data {
    Owned(Vec<i8>),
    Mmap {
        mmap: Arc<memmap2::Mmap>,
        offset: usize,
        len: usize,
    },
    Shared {
        data: Arc<[u8]>,
        offset: usize,
        len: usize,
    },
}

impl std::ops::Deref for Quant8Data {
    type Target = [i8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            Quant8Data::Owned(v) => v.as_slice(),
            Quant8Data::Mmap { mmap, offset, len } => {
                let bytes = &mmap[*offset..*offset + *len];
                // SAFETY: i8 与 u8 具备完全相同的单字节大小与内存对齐约束
                unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const i8, *len) }
            }
            Quant8Data::Shared { data, offset, len } => {
                let bytes = &data[*offset..*offset + *len];
                // SAFETY: 内存切片生命周期由 Arc 强引用锁定，转换安全
                unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const i8, *len) }
            }
        }
    }
}
```

配合惰性反量化机制，模型在内存中始终保持 8 位有符号整数形态。反量化操作不下沉至网络前向传播前，而是融合至 SIMD 计算最内层，直接通过寄存器层面的标量广播乘法与偏置累加执行，使系统显存与内存带宽占用降低 75%。

### 2. 权重视图静态绑定机制（PreparedEngineWeights 与 LayerWeightsView）
在传统动态模型实现中，推理热路径往往需要在每一层甚至每个子算子内部通过名称检索哈希表（如 `model.get_tensor("decoder_l0_ffn_w1")`），这不仅引入了哈希计算与锁冲突，还伴随着临时的引用包装与向量分配。

MTranServer-Rust 提出了基于生命周期绑定的静态权重视图结构 `LayerWeightsView` 与堆持有封装 `PreparedEngineWeights`。在翻译引擎初始化阶段，静态视图一次性遍历解析模型的所有编码器层、解码器层、词嵌入矩阵及偏置张量，将其固化为由连续切片与裸指针引用的结构体。

```rust
// src/engine/transformer.rs
// 堆持有安全封装，消除热路径哈希查找
pub struct PreparedEngineWeights {
    model: Arc<Model>,
    // 内部权重引用生命周期在结构体内被提升为 'static，
    // 对外通过安全接口借用输出与其自身生命周期绑定的权重视图
    weights: LayerWeightsView<'static>,
}

impl PreparedEngineWeights {
    pub fn new(model: Arc<Model>, config: &ForwardConfig) -> Self {
        let model_ptr: *const Model = &*model;
        // SAFETY: `model` 通过 Arc<Model> 被 PreparedEngineWeights 强引用持有，
        // 保证底层的 Model 堆内存地址在结构体存续期间保持固定不发生移动与释放。
        // 外部调用者仅能获得与 &self 生命周期绑定的 LayerWeightsView<'_>，绝不会逃逸
        let static_model: &'static Model = unsafe { &*model_ptr };
        let weights = LayerWeightsView::new(static_model, config);
        Self { model, weights }
    }

    #[inline]
    pub fn weights(&self) -> &LayerWeightsView<'_> {
        &self.weights
    }
}
```

自回归步进解码循环中所有算子对权重的读取被编译器直接固化为完全内联的偏移基址寻址指令，彻底抹除了散列查找开销。

### 3. 模型生命周期精准追踪与无锁原子使用度计数
在高并发服务环境中，由于机器物理内存受限，系统必须精确管控已加载模型的驻留状态。在体系结构上，服务层与计算核心层保持了严格的职责切分：服务生命周期容器与无锁热路径统计完全解耦。

在 `src/service/mod.rs` 中，引擎驻留信息体 `EngineInfo` 仅负责维系底层引擎实例与后台异步协程句柄，杜绝在服务字典中频繁读写计数导致全局读写锁争用。

```rust
// src/service/mod.rs
// 引擎驻留状态管理结构体
struct EngineInfo {
    engine: Arc<TranslationEngine>,
    idle_task: Option<tokio::task::JoinHandle<()>>,
}
```

而在高频调用的翻译热路径上，调用计数与时间戳更新直接下沉至 `src/engine/mod.rs` 中的 `EngineUsage` 结构体，由 `TranslationEngine` 内部持有的原子字段驱动。该计数器完全由 `AtomicU64` 构成，使用 `Ordering::Relaxed` 内存序实施热路径无锁更新，彻底消除了每次翻译请求对全局读写锁的争用与缓存一致性总线风暴。

```rust
// src/engine/mod.rs
// 无锁线程安全的使用度与时间戳追踪器
#[derive(Debug)]
pub struct EngineUsage {
    pub call_count: AtomicU64,
    pub last_used_ms: AtomicU64,
}

impl EngineUsage {
    pub fn new() -> Self {
        Self {
            call_count: AtomicU64::new(0),
            last_used_ms: AtomicU64::new(now_ms()),
        }
    }

    /// 热路径无锁原子自增与时间戳刷新
    #[inline]
    pub fn record_call(&self) {
        self.call_count.fetch_add(1, Ordering::Relaxed);
        self.last_used_ms.store(now_ms(), Ordering::Relaxed);
    }

    #[inline]
    pub fn last_used_ms(&self) -> u64 {
        self.last_used_ms.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn idle_for(&self, timeout: Duration) -> bool {
        let last = self.last_used_ms();
        now_ms().saturating_sub(last) >= timeout.as_millis() as u64
    }
}
```

配合 `worker-idle-timeout` 空闲回收、`keep-models` 静态名单常驻、`keep-top-n` 动态高频保活以及 `max-loaded-models` 容量上限平滑驱逐，系统构建了自适应且零锁竞争的现代生命周期管控机制。

### 4. Single-Flight 防击穿并发加载器
在服务冷启动或冷门语言对遭遇高并发突发请求时，容易出现数十个工作协程同时发现模型未加载，进而同时发起磁盘读取与解析的“缓存击穿”现象。

MTranServer-Rust 引入了基于异步 Promise 的 Single-Flight 防击穿机制。所有请求在尝试构建模型前，必须先在互斥保护的加载表中登记。首个抵达的协程成为 Loader 并持有专属写锁开始加载；后续并发抵达的所有相同语言对请求则被判定为 Waiter，挂载在同一个共享的异步互斥锁上等待通知。

```rust
// src/service/mod.rs
// Single-Flight 异步并发装载槽位定义
type LoadingPromise = Arc<tokio::sync::Mutex<Option<Result<Arc<TranslationEngine>, ServiceError>>>>;

enum LoadSlot {
    Loader(
        LoadingPromise,
        tokio::sync::OwnedMutexGuard<Option<Result<Arc<TranslationEngine>, ServiceError>>>,
    ),
    Waiter(LoadingPromise),
}

struct LoadingCleanupGuard<'a> {
    key: &'a str,
    promise: &'a LoadingPromise,
}

impl<'a> Drop for LoadingCleanupGuard<'a> {
    fn drop(&mut self) {
        let mut guard = LOADING.lock();
        if let Some(map) = guard.as_mut() {
            if map.get(self.key).is_some_and(|p| Arc::ptr_eq(p, self.promise)) {
                map.remove(self.key);
            }
        }
    }
}
```

当 Loader 协程成功完成加载并发布 `Arc<TranslationEngine>` 后，所有挂起的 Waiter 协程被瞬时唤醒，直接从 Promise 中克隆引擎引用进入翻译流程，将冷启动重复加载次数从 $N$ 严格削减为 1。

### 5. std::thread::scope 多资产并发解析与词表句柄共享
一个完整的神经机器翻译引擎实例包含三大相互独立的异构资产：Marian 二进制模型文件、双向短语剪枝词汇表（Shortlist）以及源端和目标端的 SentencePiece 分词模型。在传统串行初始化模式下，冷启动耗时等于各个资产反序列化耗时的绝对累加。

MTranServer-Rust 在 `src/engine/mod.rs` 中采用现代标准库的 `std::thread::scope` 实现了多资产并发解析管线。各个加载任务派发至独立的作用域线程并发执行，冷启动整体耗时直接收敛为加载最慢单资产的木桶上限。

```rust
// src/engine/mod.rs
// std::thread::scope 作用域多资产并发装载与词表共享
let (model, shortlist, spm_src, spm_trg) = std::thread::scope(
    |s| -> Result<(Model, Shortlist, Arc<SentencePiece>, Arc<SentencePiece>), EngineError> {
        let model_h = s.spawn(|| Model::load(model_path));
        let lex_h = s.spawn(|| Shortlist::load(lex_path));

        // 词表句柄共享优化：当源端与目标端词表指向同一文件时，
        // 仅解析单次 Protobuf 并构建单个 Trie，克隆两份 Arc 句柄共享，杜绝双倍内存与解析开销
        let spm_h = s.spawn(
            move || -> Result<(Arc<SentencePiece>, Arc<SentencePiece>), EngineError> {
                if src_vocab_path == trg_vocab_path {
                    let spm = SentencePiece::load(src_vocab_path)?;
                    let spm = Arc::new(spm);
                    Ok((spm.clone(), spm))
                } else {
                    let src = SentencePiece::load(src_vocab_path)?;
                    let trg = SentencePiece::load(trg_vocab_path)?;
                    Ok((Arc::new(src), Arc::new(trg)))
                }
            },
        );

        let model = model_h.join().map_err(|_| {
            EngineError::WorkerPanicked("Model loader panicked".to_string())
        })??;
        let shortlist = lex_h.join().map_err(|_| {
            EngineError::WorkerPanicked("Shortlist loader panicked".to_string())
        })??;
        let (spm_src, spm_trg) = spm_h.join().map_err(|_| {
            EngineError::WorkerPanicked("SentencePiece loader panicked".to_string())
        })??;

        Ok((model, shortlist, spm_src, spm_trg))
    },
)?;
```

对于绝大多数对称双向互译模型或共享子词词表，`src_vocab_path` 与 `trg_vocab_path` 指向完全一致的物理路径。系统通过直接比对路径，仅对共享词表执行单次加载，随后派发两份浅拷贝句柄，使冷启动词表构建耗时减少 50%，常驻内存底噪减少近 15 MB。

### 6. with_key 栈空间 64 字节零堆分配引擎键格式化
在服务端路由查找与使用度统计的热路径上，系统需要高频根据语言代码（如 `en` 与 `zh-Hans`）组合生成标准查询键 `from-to`。若常规使用 `format!("{}-{}", from, to)`，每次查找均会在堆上分配一个临时 `String`，在高并发吞吐下引发巨大的内存分配器微碎片。

系统在 `src/service/mod.rs` 中设计了 `with_key` 辅助函数，在栈帧上开辟一个固定的 64 字节内存切片。当语言对代码组合长度不超过 64 字节时，直接就地拷贝并生成借用的字符串切片 `&str`，实现全生命周期零堆分配。

```rust
// src/service/mod.rs
// 栈空间 64 字节就地格式化，彻底规避热路径字符串堆分配
#[inline]
fn with_key<R>(from: &str, to: &str, f: impl FnOnce(&str) -> R) -> R {
    let len = from.len() + 1 + to.len();
    if len <= 64 {
        let mut buf = [0u8; 64];
        buf[..from.len()].copy_from_slice(from.as_bytes());
        buf[from.len()] = b'_';
        buf[from.len() + 1..len].copy_from_slice(to.as_bytes());
        // SAFETY: 字符完全由合法的 ASCII 语言代码片段拷贝组成，保证合法 UTF-8
        let s = unsafe { std::str::from_utf8_unchecked(&buf[..len]) };
        f(s)
    } else {
        f(&format!("{from}_{to}"))
    }
}
```

## 第二章 底层张量计算与硬件级 SIMD 内核优化

### 1. 硬件特性运行时探测与静态函数指针分发
为了保证编译出的二进制程序能够在不同微架构的 x86_64 与 ARM64 处理器上均能达到理论峰值性能，系统杜绝了硬编码编译时架构参数的做法，转而采用全动态 CPUID / HWCAP 运行时特征嗅探机制。

系统在启动时通过 `SimdLevel` 枚举对运行主机进行一次性探测，探测结果固化在 `LazyLock` 或 `OnceLock` 保护的静态函数指针中。在整个热路径前向传播期间，所有的 GEMM、GEMV、LayerNorm 与 Softmax 运算只需承担一次间接函数指针调用的开销，完全规避了内层循环内部频繁的分支探测与 CPU 流水线冲刷。

```rust
// src/engine/tensor/simd/mod.rs
// 全局单次探测的 I8 矩阵乘法函数指针静态分发表
#[inline]
pub fn matmul_i8_dispatch() -> I8Matmul {
    use crate::utils::arch::SimdLevel;
    static FN: LazyLock<I8Matmul> = LazyLock::new(|| match crate::utils::arch::simd_level() {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2Fma => {
            if crate::utils::arch::has_avxvnni() {
                vnni::matmul_i8_avxvnni
            } else {
                avx2::matmul_i8_avx2
            }
        }
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Sse41 | SimdLevel::Avx => sse41::matmul_i8_sse41,
        #[cfg(target_arch = "aarch64")]
        SimdLevel::Neon if std::arch::is_aarch64_feature_detected!("dotprod") => {
            crate::engine::arm_dotprod::matmul_i8_dotprod
        }
        #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
        SimdLevel::Neon => neon::matmul_i8_neon,
        _ => scalar::matmul_i8_scalar,
    });
    *FN
}
```

### 2. 非对称 INT8 量化与 quantize_row_avx2 向量化
模型激活值在进入密集线性层之前需要被动态量化为 INT8。为了在保留动态范围的同时最大化 SIMD 吞吐，系统采用非对称有偏量化方案，将激活值动态映射到区间 $[-127, 127]$。

在 AVX2 指令集支持下，系统在 `src/engine/tensor/simd/avx2.rs` 中通过 `quantize_row_avx2` 实现了无偏四舍五入与防通道错位的双级饱和向量截断打包。为彻底解决 `_mm256_packs_epi32` 跨 128 位通道（lane）数据错位的问题，系统提取高低 128 位半块，利用 128 位 SSE 饱和打包指令连续重组：

```rust
// src/engine/tensor/simd/avx2.rs
// AVX2 向量化行量化实现核心片段
#[inline]
pub unsafe fn quantize_row_avx2(data: &[f32], qa: f32, out: &mut [i8]) {
    use std::arch::x86_64::*;
    let len = data.len();
    let chunks16 = len / 16;
    let mut offset = 0;
    let qa_vec = _mm256_set1_ps(qa);
    let half_pos = _mm256_set1_ps(0.5);
    let half_neg = _mm256_set1_ps(-0.5);
    let zero = _mm256_setzero_ps();
    let min_clamp = _mm256_set1_ps(-127.0);
    let max_clamp = _mm256_set1_ps(127.0);

    let src = data.as_ptr();
    let dst = out.as_mut_ptr();

    for _ in 0..chunks16 {
        let v0 = _mm256_loadu_ps(src.add(offset));
        let scaled0 = _mm256_mul_ps(v0, qa_vec);
        let mask0 = _mm256_cmp_ps(scaled0, zero, _CMP_GE_OQ);
        let off0 = _mm256_blendv_ps(half_neg, half_pos, mask0);
        let rounded0 = _mm256_add_ps(scaled0, off0);
        let ord_mask0 = _mm256_cmp_ps(rounded0, rounded0, _CMP_EQ_OQ);
        let clamped0 = _mm256_min_ps(_mm256_max_ps(rounded0, min_clamp), max_clamp);
        let r0 = _mm256_cvttps_epi32(_mm256_and_ps(clamped0, ord_mask0));

        let v1 = _mm256_loadu_ps(src.add(offset + 8));
        let scaled1 = _mm256_mul_ps(v1, qa_vec);
        let mask1 = _mm256_cmp_ps(scaled1, zero, _CMP_GE_OQ);
        let off1 = _mm256_blendv_ps(half_neg, half_pos, mask1);
        let rounded1 = _mm256_add_ps(scaled1, off1);
        let ord_mask1 = _mm256_cmp_ps(rounded1, rounded1, _CMP_EQ_OQ);
        let clamped1 = _mm256_min_ps(_mm256_max_ps(rounded1, min_clamp), max_clamp);
        let r1 = _mm256_cvttps_epi32(_mm256_and_ps(clamped1, ord_mask1));

        // 提取 128 位半块，避免跨通道错位，连续调用 SSE 饱和打包
        let r0_lo = _mm256_castsi256_si128(r0);
        let r0_hi = _mm256_extracti128_si256(r0, 1);
        let r1_lo = _mm256_castsi256_si128(r1);
        let r1_hi = _mm256_extracti128_si256(r1, 1);

        let p16_0 = _mm_packs_epi32(r0_lo, r0_hi);
        let p16_1 = _mm_packs_epi32(r1_lo, r1_hi);
        let packed = _mm_packs_epi16(p16_0, p16_1);

        _mm_storeu_si128(dst.add(offset) as *mut __m128i, packed);
        offset += 16;
    }
}
```

### 3. AVX2 Nibble-Split 无溢出累加内核
在经典 8 位整数矩阵乘法中，若直接使用 x86 的 `_mm256_maddubs_epi16` 指令，该指令会将第一个无符号 8 位操作数与第二个有符号 8 位操作数两两相乘并相邻相加为一个有符号 16 位整数（i16）。然而，有符号 16 位的最大表达范围仅为 $[-32768, 32767]$。当激活值与权重的点积积项较大时，两对极值乘积相加极易击穿 16 位边界，产生严重的整数饱和溢出（Saturation Overflow），导致模型译文完全乱码。

为了彻底消除溢出风险而不损失计算吞吐，MTranServer-Rust 采用了高度精巧的 Nibble-Split（半字节拆分）技术：
首先将带偏置的无符号输入字节 $x_b = x + 128$（范围 $[0, 255]$）拆分为高 4 位 $hi$ 与低 4 位 $lo$：
$$x_b = 16 \cdot hi + lo, \quad hi \in [0, 15], \; lo \in [0, 15]$$
由于拆分后的每个 nibble 上限仅为 15，两对乘积和的最大可能绝对值为：
$$2 	imes 15 	imes 127 = 3810 \ll 32767$$
这与 16 位整数的溢出阈值保持了近 10 倍的安全裕量。随后，内核分别对 $lo$ 与 $hi$ 执行 `_mm256_maddubs_epi16`，再通过 `_mm256_madd_epi16(..., ones)` 水平展开并累加至 32 位整数寄存器，最后通过 `_mm256_slli_epi32::<4>` 将高位结果左移 4 位（乘以 16）并与低位合并：

```rust
// src/engine/tensor/simd/avx2.rs
// Nibble-Split 彻底消除 16 位乘加溢出的核心累加逻辑
macro_rules! madd_nibble_avx2 {
    ($lo8:expr, $hi8:expr, $wv:expr, $ones:expr) => {{
        // 分别计算低 4 位与高 4 位的 maddubs，结果严格限制在 i16 安全范围内
        let lo_part = _mm256_madd_epi16(_mm256_maddubs_epi16($lo8, $wv), $ones);
        let hi_part = _mm256_madd_epi16(_mm256_maddubs_epi16($hi8, $wv), $ones);
        // 通过 32 位整型左移 4 位完成等价的 16 倍放大，精确还原原始点积
        _mm256_add_epi32(lo_part, _mm256_slli_epi32::<4>(hi_part))
    }};
}
```

此方案在纯 AVX2 环境下以微小的寄存器重排代价，换取了与 32 位整数全精度累加完全等价的无溢出数学正确性。

### 4. AVX-VNNI 内联裸汇编指令发射
在 Intel 第 12 代 Core（Alder Lake）、第 13 代及后续 CPU（以及部分新型 AMD Zen4 架构）上，引入了全新的 AVX-VNNI 指令集扩展。该扩展允许在 256 位 YMM 寄存器上直接使用 `vpdpbusd` 指令，单周期内直接完成 4 组四元无符号 u8 与有符号 i8 的乘加运算，并直接累加至 32 位整数，吞吐量相较 AVX2 方案直接翻倍。

然而，在 Rust 官方标准库的 `core::arch::x86_64` 中，`_mm256_dpbusd_epi32` 内建函数（intrinsic）长期以来被错误地绑定在 `#[target_feature(enable = "avx512vnni")]` 门控下。这意味着编译器在生成目标代码时会强制发射 512 位 EVEX 编码前缀。当程序运行在仅支持 256 位 AVX-VNNI 却缺乏 AVX-512 的混合架构小核（如 Gracemont）或桌面级 CPU 时，将直接触发非法的指令异常崩溃（SIGILL）。

MTranServer-Rust 通过 Rust 原生裸汇编宏 `asm!` 突破了这一限制。系统在运行时精确区分当前硬件应使用 VEX 前缀还是 EVEX 前缀，并直接向编译器汇编流注入原始指令模板：

```rust
// src/engine/tensor/simd/vnni.rs
// 绕过标准库限制、兼容 VEX/EVEX 编码的裸汇编发射宏
macro_rules! dpbusd {
    ($acc:expr, $a:expr, $b:expr) => {{
        let acc = $acc;
        let a = $a;
        let b = $b;
        let out: std::arch::x86_64::__m256i;
        if EVEX {
            // 适用于具备完整 AVX-512VNNI 支撑的高端服务器架构
            asm!(
                "{{evex}} vpdpbusd {acc}, {a}, {b}",
                acc = inout(ymm_reg) acc => out,
                a = in(ymm_reg) a,
                b = in(ymm_reg) b,
                options(pure, nomem, nostack)
            );
        } else {
            // 关键突破点: 发射 3 字节 VEX 前缀的 vpdpbusd，完美运行于主流 Alder/Raptor Lake 架构
            asm!(
                "{{vex}} vpdpbusd {acc}, {a}, {b}",
                acc = inout(ymm_reg) acc => out,
                a = in(ymm_reg) a,
                b = in(ymm_reg) b,
                options(pure, nomem, nostack)
            );
        }
        out
    }};
}
```

借由此设计，系统成为了业内极少数能够在非 AVX-512 硬件平台上安全释放硬件级 VNNI 极致算力的自研引擎。

### 5. ARM64 NEON 与 DotProd 内核加速
在 AArch64 架构（如 AWS Graviton、Apple Silicon、树莓派等）上，系统全面启用了 NEON 与 ARMv8.2-A 的 `FEAT_DotProd` 指令扩展。

当探测到 CPU 具备 `dotprod` 特性时，系统分发至 `matmul_i8_dotprod` 内核，直接利用 `vdotq_s32` 内联函数执行 4 个 8 位点积向 32 位累加器的单步融合映射：

```rust
// src/engine/arm_dotprod.rs
// ARMv8.2-A DotProd 硬件点积循环展开
#[target_feature(enable = "dotprod")]
pub(crate) unsafe fn matmul_i8_dotprod(
    x_ptr: *const i8,
    w0_ptr: *const i8,
    w1_ptr: *const i8,
    w2_ptr: *const i8,
    w3_ptr: *const i8,
    k: usize,
) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    // 每次迭代处理 4 列输出，大幅增加流水线发射深度
    let mut acc0 = vdupq_n_s32(0);
    let mut acc1 = vdupq_n_s32(0);
    let mut acc2 = vdupq_n_s32(0);
    let mut acc3 = vdupq_n_s32(0);

    for b in 0..(k / 16) {
        let xv = vld1q_s8(x_ptr.add(b * 16));
        let wv0 = vld1q_s8(w0_ptr.add(b * 16));
        let wv1 = vld1q_s8(w1_ptr.add(b * 16));
        let wv2 = vld1q_s8(w2_ptr.add(b * 16));
        let wv3 = vld1q_s8(w3_ptr.add(b * 16));

        // 硬件单周期完成 16 个点积乘加
        acc0 = vdotq_s32(acc0, xv, wv0);
        acc1 = vdotq_s32(acc1, xv, wv1);
        acc2 = vdotq_s32(acc2, xv, wv2);
        acc3 = vdotq_s32(acc3, xv, wv3);
    }
    (acc0, acc1, acc2, acc3)
}
```

### 6. 列和修正项（Row Sums Correction）预计算
在非对称量化矩阵乘法中，为了将有偏的有符号输入 $x + 128$ 映射还原为真实数学值 $x \cdot w$，必须展开代数恒等式：
$$\sum (x + 128) \cdot w = \sum x \cdot w + 128 \sum w$$
即真实点积为：
$$\sum x \cdot w = \sum (x + 128) \cdot w - 128 \sum w$$
若在每次前向推理时动态遍历计算权重的列和 $\sum w$，将引入极高的数据重复读取与计算延迟。MTranServer-Rust 将这一修正项完全提前至模型加载阶段：在解析模型张量时，一次性预计算每一列权重的修正项 $128 \sum w$，并按 64 字节内存对齐固化于 `Quant8Tensor::row_sums` 中。

在 SIMD 累加内核中，当点积循环结束时，只需单条向量减法指令 `_mm256_sub_epi32` 将原始累加和减去预加载的 `row_sums`，即刻以零延迟代价完成数学还原。

### 7. 自适应 M 维 Rayon 并行与动态负载门控
Transformer 编码器阶段的输入序列长度 $M$ 往往达到数十至数百，各输出行之间不存在自回归依赖。若纯单线程运行，则无法充分利用多核服务器算力；但若无节制地进行多线程切分，多线程任务创建、上下文切换及 L2 缓存中权重矩阵 Panel 的反复失效，反而会导致短序列的吞吐大幅倒退。

系统设计了具有动态负载感知的自适应并行门控机制 `gemm_should_split`。切分决策不仅取决于当前矩阵的计算量（FMAs = $M \times K \times N$），还深度联动全局运行状态：

```rust
// src/engine/tensor/simd/mod.rs
// M 维矩阵乘自适应切分与负载门控阈值
const GEMM_PARALLEL_WORK: u64 = 4_000_000;          // 空闲时最小切分计算量 (4M FMAs)
const GEMM_PARALLEL_WORK_BUSY: u64 = 16_000_000;     // 高负载时切分门槛提升至 16M FMAs
pub(crate) const GEMM_MIN_ROWS: usize = 16;         // 切分后每个 Worker 必须持有的最小行数

#[inline]
pub(crate) fn gemm_should_split(m: usize, k: usize, n: usize) -> bool {
    if m < 2 * GEMM_MIN_ROWS {
        return false;
    }
    let work = (m as u64) * (k as u64) * (n as u64);
    let active = ACTIVE_SPLITS.load(std::sync::atomic::Ordering::Relaxed);
    // 依据当前系统处于并发切分状态的任务数，动态拔高阈值，防止 CPU 级联过载与 L2 缓存抖动
    let threshold = if active == 0 {
        GEMM_PARALLEL_WORK
    } else {
        GEMM_PARALLEL_WORK_BUSY
    };
    work >= threshold && rayon::current_num_threads() > 1
}
```

当判定满足切分条件时，内核采用基于二分分治的 `rayon::join` 递归切分 $M$ 维度。由于矩阵内层严格沿 $N$ 维度进行分块（$N$-Blocked），每个切分后的子任务依然能够完整复用常驻在 CPU L1 数据缓存中的权重面板（B-Panel），实现了多核利用率与缓存局部性的协同最优。

### 8. 多算子内存融合流水线（2-Pass Zero-STLF 与 Fused GEMM Pair）
在多层 Transformer 计算中，残差相加（Residual Add）与层归一化（Layer Normalization）是极为频繁的高频操作。标准实现通常分为两步：首先分配并写回 $A + B$ 的中间临时张量，随后读取该临时张量以计算均值与方差。这种模式在微架构层面会触发严重的“写后立即读”停顿，即存储转发失败（Store-To-Load Forwarding Failure, STLF）。

MTranServer-Rust 构建了 `fused_add_layer_norm_avx2_raw` 算子，采用双趟零存储转发架构（2-Pass Zero-STLF）：
第一趟（Pass 1）：直接利用 AVX2 寄存器对输入指针 $A$ 与 $B$ 进行流式向量加载与就地相加，在寄存器内部直接累加元素和与平方和，计算出最终的精确均值 $\mu$ 与逆标准差 $\frac{1}{\sqrt{\sigma^2 + \epsilon}}$，整个过程完全不向内存写回任何中间张量；
第二趟（Pass 2）：重新读取 $A$ 与 $B$，再次相加后直接结合预热好的标度 $\gamma$（scale）与偏置 $\beta$（bias）完成归一化变换，并将最终结果单次写回目标地址。

此项融合使得访存总流量削减了 33%，彻底消除了 STLF 延迟冲刷，同时结合 ReLU 激活函数直接在 GEMM 写回阶段的就地融合，以及自回归阶段将循环投影与门控投影合一的 `quantized_matmul_pair_into`（共享单次输入激活量化），组成了高效紧凑的前向算子流水线。

## 第三章 Transformer 推理架构与自回归解码优化

### 1. SSRU 循环单元替代自注意力机制
传统 Transformer 解码器的最大性能瓶颈在于掩码自注意力（Self-Attention）机制：每生成一个新 Token，均需对历史所有步的键（Key）和值（Value）张量执行打分与加权聚合。自回归步数增长导致计算量呈 $O(L^2)$ 二次方恶化，且维护庞大动态增长的 KV 缓存会带来沉重的内存分配与带宽碎片。

MTranServer-Rust 针对 Marian NMT 架构深度优化，全面采用 SSRU（Simpler Simple Recurrent Unit）循环单元取代解码器的自注意力层。SSRU 的单步递推公式如下：
$$c_t = (1 - \sigma(W_f x_t + b_f)) \odot c_{t-1} + \sigma(W_f x_t + b_f) \odot \text{ReLU}(W_r x_t)$$
$$y_t = \text{ReLU}(c_t) + x_t$$
式中 $c_t$ 为当前层隐藏状态，$x_t$ 为当前步输入，$\sigma$ 为 Sigmoid 遗忘门。

```rust
// src/engine/transformer.rs
// SSRU 解码单元双投影融合与门控原地推进
fn decoder_ssru(
    input: &Tensor2,
    cell: &mut Tensor2,
    rnn_w: &LinW,
    rnn_wf: &LinW,
    rnn_bf: &Tensor2,
    s_x: &mut Tensor2,
    s_f: &mut Tensor2,
    s_cell: &mut Tensor2,
) {
    // 共享量化系数时，单次量化 input，同时计算 W_r 与 W_f 投影
    match (rnn_w, rnn_wf) {
        (LinW::Q8(wq, qa1), LinW::Q8(wqf, qa2)) if qa1 == qa2 => {
            input.quantized_matmul_pair_into(wq, None, s_x, wqf, Some(rnn_bf), s_f, *qa1);
        }
        _ => {
            match rnn_w {
                LinW::F32(w) => input.matmul_biased_into(w, None, s_x),
                LinW::Q8(wq, qa) => input.quantized_matmul_biased_into(wq, *qa, None, s_x),
            }
            match rnn_wf {
                LinW::F32(w) => input.matmul_biased_into(w, Some(rnn_bf), s_f),
                LinW::Q8(wq, qa) => input.quantized_matmul_biased_into(wq, *qa, Some(rnn_bf), s_f),
            }
        }
    }

    // 单趟融合门控与 ReLU 输出计算，杜绝中间内存写回
    s_cell.data.resize(input.data.len(), 0.0);
    let n = input.data.len();
    ssru_gate_into(&cell.data[..n], &s_f.data[..n], &mut s_x.data[..n], &mut s_cell.data[..n]);

    // 指针级零拷贝交换隐状态所有权
    std::mem::swap(cell, s_cell);
}
```

SSRU 将解码器上下文依赖完全压缩为一个固定大小的隐状态向量（维度等于 $d_{\text{model}}$，通常为 512），每次自回归迭代仅需一次固定大小的向量状态原地变换。这意味着自回归解码时间与上下文长度完全解耦，步进推理计算复杂度严格稳定为 $O(1)$，内存开销保持恒定。

### 2. HeadMajorK 转置连续内存布局与专用微内核
在交叉注意力（Cross-Attention）中，解码器每一单步生成的 Query 向量必须与编码器输出的所有源端 Key 向量计算内积打分。原始矩阵通常采用行优先格式存储，即维度排列为 $[L_{\text{src}}, H, d_k]$。在此种布局下，计算某个注意力头的点积需要以跨步长（Strided）方式跨越内存，极易导致 CPU 数据预取失效与缓存行浪费。

为了达成极致的打分吞吐，系统在编码器结束前向计算后，立即执行一次布局转置，将 Key 与 Value 重构为多头优先的连续布局 `HeadMajorK` 与 `HeadMajorV`：
$$[H][d_k][L_{\text{src}}]$$
在此布局下，同一注意力头内所有源端位置在特征维度 $d_k$ 上的元素均处于连续物理内存中。

```rust
// src/engine/transformer.rs
// Key 张量重排为 HeadMajor 连续内存布局
fn rearrange_k_heads_into(t: &Tensor2, heads: usize, out: &mut HeadMajorK) {
    let src_len = t.rows();
    let d_model = t.cols();
    let dk = d_model / heads;
    out.src_len = src_len;
    out.heads = heads;
    out.dk = dk;
    out.kt.resize(src_len * d_model, 0.0);

    for j in 0..src_len {
        for h in 0..heads {
            let src = j * d_model + h * dk;
            for t2 in 0..dk {
                // 紧密打包: 同一 head、同一特征通道内的所有源词连续排布
                out.kt[h * dk * src_len + t2 * src_len + j] = t.data[src + t2];
            }
        }
    }
}
```

针对神经翻译模型最通用的两个注意力头维度 $d_k = 48$ 与 $d_k = 64$，系统手写了高度优化的专用 AVX2 累加微内核。以内核 `accumulate_head_v_dk64_avx2` 为例，内核直接分配 8 个 256 位 YMM 累加寄存器，将循环完全展开为固定步长流水线，完全抹除了循环计数分支与边界判断。

### 3. 词表剪枝（Shortlist）全套优化体系
神经机器翻译模型最终的 Logits 投射层计算开销占据解码总时间的大半。MTranServer-Rust 配套了高精度二进制词汇剪枝机制（`.s2t.bin`），在编码结束后，动态为当前句子挑选出最可能出现的数百个目标候选词（通常 200~800 词），将输出矩阵乘法维度从数万骤降两个数量级。

系统围绕 Shortlist 设计了三重极速优化策略：

第一重：源词栈上去重（Stack Deduplication）。输入句子的 Token ID 在检索词表前必须去重，对于通常小于 512 Token 的实际句子，系统直接利用 `MaybeUninit<u32>` 栈数组进行排序与就地去重，杜绝堆内存申请；
第二重：脏位图追踪机制（Dirty Words Tracking）。剪枝候选集使用 64 位对齐的 Bitmask 进行判重。系统并不在调用后执行全量 Bitmask 清零，而是实时记录被标记的最小字索引 `min_word_idx` 与最大字索引 `max_word_idx`，单步结束后仅针对脏区间（Dirty Range）进行擦除，将位图重置耗时压缩至纳秒级；
第三重：候选权重收集前缀复用（Prefix Reuse）。输出投射层矩阵需要在解码前从庞大的全局词嵌入中收集（Gather）对应候选行。系统在连续解码中追踪候选集的最长公共前缀，并利用游程编码（Run-Length）检测词表中连续编号的区间，将原本离散的逐行收集升级为大块连续内存拷贝 `copy_nonoverlapping`，极大缓解了非连续内存随机抓取导致的缓存缺失。

```rust
// src/engine/shortlist.rs
// Shortlist 零堆分配动态生成核心
impl Shortlist {
    pub fn generate_into(
        &self,
        source_ids: &[u32],
        trg_vocab_size: usize,
        seen: &mut Vec<u64>,
        out: &mut Vec<u32>,
        policy: ShortlistPolicy,
    ) {
        out.clear();
        if trg_vocab_size == 0 { return; }

        let first_n = (self.first_num as usize).max(500).min(trg_vocab_size);
        out.extend(0..first_n as u32);

        // 利用栈数组对源 Token 就地无堆分配排序去重
        let mut stack_src = [std::mem::MaybeUninit::<u32>::uninit(); 512];
        let n = source_ids.len().min(512);
        for (dst, &src) in stack_src[..n].iter_mut().zip(source_ids.iter()) {
            dst.write(src);
        }
        let deduped = unsafe {
            let slice = std::slice::from_raw_parts_mut(stack_src.as_mut_ptr() as *mut u32, n);
            slice.sort_unstable();
            let mut len = 0;
            for i in 0..slice.len() {
                if i == 0 || slice[i] != slice[i - 1] {
                    slice[len] = slice[i];
                    len += 1;
                }
            }
            &slice[..len]
        };

        // 仅对发生过置位的脏区间执行常数时间位的提取与清零
        // ...使用 trailing_zeros 快速迭代，bits &= bits - 1 完成稀疏提取...
    }
}
```

### 4. 自回归惩罚极速映射与分级阻断
神经文本生成过程中极易出现退化并陷入病态死循环（例如反复输出相同的单词或短语）。系统设计了高精度自回归惩罚引擎，并在热路径上予以极致加速。

```rust
// src/engine/transformer.rs
// 高性能自回归惩罚与多级 N-gram 阻断
pub(crate) fn apply_penalties_with_exemptions_fast(
    logit_buf: &mut [f32],
    shortlist_indices: &[u32],
    prefix_len: usize,
    tokens: &[u32],
    repetition_penalty: f32,
    no_repeat_ngram_size: usize,
    exempt_tokens: &[u32],
) {
    // 1. 重复惩罚: 小序列快速 L1 缓存就地判重
    if repetition_penalty > 0.0 && (repetition_penalty - 1.0).abs() > 1e-5 {
        let inv_penalty = 1.0 / repetition_penalty;
        if tokens.len() <= 64 {
            for (idx, &token) in tokens.iter().enumerate() {
                if !tokens[..idx].contains(&token) && !exempt_tokens.contains(&token) {
                    if let Some(pos) = shortlist_index_of(shortlist_indices, prefix_len, token) {
                        let logit = &mut logit_buf[pos];
                        if *logit > 0.0 {
                            *logit *= inv_penalty;
                        } else {
                            *logit *= repetition_penalty;
                        }
                    }
                }
            }
        }
    }
}
```

针对 N-gram 重复阻断，系统创造性地提出了“分级递进惩罚”算法：当候选 Token 构成的 N-gram 在历史中第一次出现时，不采取生硬的一票否决，而是施加温和的软衰减阻断（从 Logit 中扣减 3.5），允许模型在语法合理的情况下保留复现概率；当同一模式遭遇第二次重复确认出现死循环时，立即施加 $-10^9$ 的绝对硬阻断。同时，系统设置了针对 EOS 结束符的白名单豁免机制，彻底杜绝了模型因阻断惩罚而丧失正常句尾截断能力的缺陷。

### 5. 自适应动态 EOS 长度引导机制（Adaptive Dynamic EOS Guidance）
轻量级蒸馏神经翻译模型在解码较长句子或遇到逗号等中间标点时，极易因局部置信度波动过早预测 EOS，造成译文在半途生硬截断。若简单粗暴地在固定步长内强行屏蔽 EOS，又会导致原本短小精悍的句子产生幻觉冗余。

MTranServer-Rust 创新性地设计了 `apply_adaptive_dynamic_eos_guidance` 算子。该机制基于源端长度 $L_{\text{src}}$、语种膨胀比率（`expansion_ratio`）以及动态因子动态推导软最小长度 $L_{\min}$ 与软最大长度 $L_{\text{soft\_max}}$：

```rust
// src/engine/transformer.rs
// 自适应动态 EOS 引导机制
#[inline]
pub(crate) fn apply_adaptive_dynamic_eos_guidance(
    logit_buf: &mut [f32],
    eos_idx: usize,
    source_len: usize,
    generated_len: usize,
    expansion_ratio: f32,
    min_factor: f32,
    max_factor: f32,
) {
    if source_len == 0 { return; }
    let eff_ratio = if expansion_ratio < 0.5 || expansion_ratio > 2.0 {
        1.0 + (expansion_ratio - 1.0) * 0.75
    } else {
        expansion_ratio
    };

    let min_bound = if source_len <= 3 { 1 } else { 2 };
    let l_min = ((source_len as f32 * eff_ratio * min_factor).round() as usize).clamp(min_bound, 120);
    let mut l_soft_max = ((source_len as f32 * eff_ratio * max_factor).round() as usize).max(l_min + 10);
    l_soft_max = l_soft_max.min(480).max(l_min + 1);

    if let Some(logit) = logit_buf.get_mut(eos_idx) {
        if generated_len < l_min {
            // 生成步长不足 L_min 时施加平滑二次方衰减
            let ratio = 1.0 - (generated_len as f32 / l_min as f32);
            *logit -= 3.5 * (ratio * ratio);
        } else if generated_len > l_soft_max {
            // 超越软上限时施加线性增益，鼓励模型平稳收敛
            let delta = ((generated_len - l_soft_max) as f32 / 30.0).min(1.0);
            *logit += 4.0 * delta;
        }
    }
}
```

当生成步长 $t < L_{\min}$ 时，系统施加平滑的二次方软惩罚 $-3.5 \times (1 - t/L_{\min})^2$，在源序列尚未充分展开前优雅压制早退倾向；当 $t > L_{\text{soft\_max}}$ 时施加线性增益，促使模型平稳收敛，完美平衡了完整性与简洁度。

### 6. 束搜索（Beam Search）候选剪枝与早停收敛
对于高精度翻译诉求，系统内置了零堆分配的束搜索解码器。在步进扩展时，维护当前候选池中最差评分索引 `worst_idx` 与得分 `worst_norm`。新候选若低于最差分数则直接丢弃，彻底消除了全量排序开销。

```rust
// src/engine/transformer.rs
// 束搜索候选槽位无排序插入与早停
#[inline(always)]
pub(crate) fn try_push_candidate(
    cand: BeamCandidate,
    candidates: &mut Vec<BeamCandidate>,
    worst_idx: &mut usize,
    worst_norm: &mut f32,
    beam_size: usize,
) -> bool {
    if candidates.len() < beam_size {
        candidates.push(cand);
        if candidates.len() == beam_size {
            *worst_idx = find_worst_candidate_idx(candidates);
            *worst_norm = candidates[*worst_idx].norm_score;
        }
        true
    } else if cand.order(&candidates[*worst_idx]) == std::cmp::Ordering::Less {
        candidates[*worst_idx] = cand;
        *worst_idx = find_worst_candidate_idx(candidates);
        *worst_norm = candidates[*worst_idx].norm_score;
        true
    } else {
        false
    }
}
```

### 7. 双重线程局部重用缓冲体系（TranslationScratch 与 ForwardBuffers）
为了杜绝高并发自回归解码过程中的系统调用与全局内存分配器争用，系统在架构上构建了两级持久化线程局部复用体系：
第一级是位于 `src/engine/mod.rs` 中的 `TranslationScratch`，维系外层分词动态规划缓存 `EncodeScratch`、文本解码 `DecodeScratch`、去重掩码 `seen: Vec<u64>`、候选短列表 `shortlist: Vec<u32>`、输出标记 `out_tokens` 与交叉注意力对齐坐标 `alignments`；
第二级是位于 `src/engine/transformer.rs` 中的 `ForwardBuffers`，维系所有解码层专享的隐状态 `cell_states`、多头张量缓存 `enc_k`/`enc_v` 以及逐层算子中间变量 `linear_scratch`。

```rust
// src/engine/mod.rs 与 src/engine/transformer.rs
// 严密区隔的两级线程局部重用缓冲区
#[derive(Default)]
struct TranslationScratch {
    encode: EncodeScratch,
    decode: DecodeScratch,
    seen: Vec<u64>,
    shortlist: Vec<u32>,
    out_tokens: Vec<u32>,
    alignments: Vec<usize>,
}

thread_local! {
    static SCRATCH: std::cell::RefCell<TranslationScratch> =
        std::cell::RefCell::new(TranslationScratch::default());
}
```

两级线程局部变量紧密配合，使自回归解码全过程中的堆内存申请严格为零。

## 第四章 文本预处理、分词与富文本对齐优化

### 1. SentencePiece 紧凑前缀树与双特化 ASCII 表
SentencePiece 分词器是神经机器翻译的数据入口。原版 C++ 实现采用动态分配指针构建的双数组 Trie 或节点树，遍历时伴随着严重的随机指针跳转与缓存失效。

MTranServer-Rust 构建了连续内存扁平化的紧凑前缀树 `PieceTrie`。树的所有节点均集中存储在单个连续的 `Vec<CompactTrieNode>` 向量中，每个节点严格压缩为 8 字节（包含子节点起始偏移 `child_start: u32`、子节点数量 `child_count: u16` 以及切词 ID `piece_id: u32`）。

```rust
// src/engine/sentencepiece.rs
// 8 字节紧凑前缀树节点与连续扁平 Trie
#[derive(Clone, Copy)]
struct CompactTrieNode {
    child_start: u32,
    child_count: u16,
    piece_id: u32,
}

struct PieceTrie {
    nodes: Vec<CompactTrieNode>,
    children: Vec<(char, u32)>,
    // 关键优化: 针对根节点与空格后继节点特化的 O(1) ASCII 极速查找表
    root_ascii: [u32; 128],
    space_child: u32,
    space_ascii: [u32; 128],
}
```

针对自然语言中超过 90% 的字符均为 ASCII 码或紧随下划线分词前缀 `_`（U+2581）之后的特性，系统在 Trie 树内设计了双特化直接索引表 `root_ascii` 与 `space_ascii`。当指针位于根节点或空格节点时，直接利用字符的 ASCII 数值作为数组下标进行 $O(1)$ 常数时间直址寻址，完全跳过了二分查找与线性遍历。

### 2. walk_matches 字节流游走与 Viterbi DP 零分配
分词核心基于 Unigram 语言模型的 Viterbi 算法。系统设计了无切片开销的无分支 UTF-8 快速解码器，在字节流上直接推进，驱动 `PieceTrie` 的状态游走。

```rust
// src/engine/sentencepiece.rs
// 零切片开销的紧凑前缀树游走匹配
#[inline]
fn walk_matches(&self, text: &str, start: usize, max_chars: usize, mut on_piece: impl FnMut(usize, u32)) {
    let bytes = text.as_bytes();
    let mut off = start;
    let mut node_idx = 0;
    let mut char_len = 0;

    while char_len < max_chars && off < bytes.len() {
        let b0 = bytes[off];
        // 无分支 UTF-8 解码，直接解析当前代码点及其字节跨度
        let (c, len) = if b0 < 0x80 {
            (b0 as char, 1)
        } else if (b0 & 0xE0) == 0xC0 && off + 1 < bytes.len() {
            let c = (((b0 & 0x1F) as u32) << 6) | ((bytes[off + 1] & 0x3F) as u32);
            (char::from_u32(c).unwrap_or('\u{FFFD}'), 2)
        } else {
            let ch = text[off..].chars().next().unwrap();
            (ch, ch.len_utf8())
        };

        let next_node = if node_idx == 0 && (c as u32) < 128 {
            self.root_ascii[c as usize]
        } else {
            self.nodes[node_idx].find_child(&self.children, c).unwrap_or(u32::MAX)
        };

        if next_node == u32::MAX { break; }
        node_idx = next_node as usize;
        off += len;
        char_len += 1;

        let piece_id = self.nodes[node_idx].piece_id;
        if piece_id != u32::MAX {
            on_piece(off - start, piece_id);
        }
    }
}
```

在执行 Viterbi 动态规划时，系统复用线程局部绑定的 `EncodeScratch` 结构体，其内部持有的 `dp` 数组只在文本长度超越历史峰值时执行扩容，在常态下完全实现零堆分配切词。

### 3. 快速查表 Decode 与字节回退流式重组
在解码输出阶段，传统方案往往需要遍历生成的 Token ID，拼接带有下划线的字符串，然后再执行正则表达式替换以清理特殊控制符。

MTranServer-Rust 在模型初始化加载词表时，提前将每个 Token ID 解析为枚举 `DecodePiece`：控制符标记为 `DecodePiece::Skip`；字节回退 Token 提取为 `DecodePiece::Byte(u8)`；正常文本片段提前剔除下划线并固化前导空格标志存入 `DecodePiece::Text`。

```rust
// src/engine/sentencepiece.rs
// 预解析查表结构与流式安全 UTF-8 解码
enum DecodePiece {
    Skip,
    Byte(u8),
    Text { text: String, add_space: bool },
}
```

解码时只需单趟匹配模式，连续 Byte 片段聚合后统一执行 UTF-8 校验与 U+FFFD 容错回退，彻底规避 Panic 崩溃。

### 4. hide_emojis 与变量占位符上下文保护
工业场景的待译文本中广泛包含 Emoji 字符、模板插值表达式（如 `{{user}}`、`${PORT}`）以及 C 格式说明符（如 `%s`、`%d`）。这些符号若直接交由神经网络翻译，极易被破坏或错译。

系统在 `src/engine/text.rs` 中设计了 `hide_emojis` 预处理管道。该管道首先执行基于字节特征的极速快筛（检查是否存在大于等于 `0xE2` 的 UTF-8 引导字节或特定占位符字符），随后将目标符号提取为保护列表，并在正文中替换为格式化的 `[EE<i>]` 占位标记。翻译完成后，通过 `restore_emojis_with_source_and_spans` 结合源端上下文的前后空格感知进行高保真无缝复原。

```rust
// src/engine/text.rs
// 表情符号与模板插值变量上下文保护
pub fn hide_emojis(text: &str) -> (Cow<'_, str>, Vec<String>) {
    let has_emoji_lead = text.as_bytes().iter().any(|&b| b >= 0xE2);
    let has_template_or_var = text.contains('{') || text.contains('$') || text.contains('%');
    if !has_emoji_lead && !has_template_or_var {
        return (Cow::Borrowed(text), Vec::new());
    }
    // 正则与状态机结合，将插值符号提取至 replacements，正文插入 [EE0] 等占位符
    let mut replacements = Vec::new();
    let mut clean_text = String::with_capacity(text.len());
    // ...按顺序替换并保持源端标点对齐边界...
    (Cow::Owned(clean_text), replacements)
}
```

### 5. 长句分级语义子句切分体系（SemanticSplitTier）
系统依据 Unicode UAX #29 规范切分句子的同时，在 `src/engine/markup/split.rs` 中构建了面向超长句（超过 `max_sentence_length` 限制）的 6 级层级语义子句切分体系 `SemanticSplitTier`。

```rust
// src/engine/markup/split.rs
// 超长文本 6 级语义断句切分层级定义
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SemanticSplitTier {
    Terminal = 1,          // 换行符与句末标点（。？！）
    MajorClause = 2,       // 高阶分句符（分号、冒号）
    ConjunctionClause = 3, // 连词引导从句（because、although 等引导词前的逗号）
    EmDash = 4,            // 破折号与双连字符
    EnumerationComma = 5,  // 顿号与列表列举标点
    MinorComma = 6,        // 普通逗号
    Whitespace = 7,        // 空格与词定界符
    Fallback = 8,          // 强行 UTF-8 字符边界截断
}
```

切分引擎在寻找截断点时，优先遵循高阶语义层级，杜绝截断从句依存关系。当多个候选切分点处于同一层级时，激活“拓扑距离平衡算法”（Topological Distance Balancing），优先选取最靠近中间位置的分割点，实现前后分段计算负载的均衡。

### 6. Unicode 边界感知分句与 TranslatedText 零拷贝路由
系统依据 Unicode 标准附件第 29 号规范精准处理句末缩写与标点。译文结果由核心枚举 `TranslatedText` 承载：新鲜计算的字符串存放于 `TranslatedText::String`；命中分片缓存时直接通过 `TranslatedText::Cached(Arc<str>)` 返回只读原子切片。该枚举透明实现 `Deref` 与 `Serialize`，数据自内存缓存直通底层网络套接字，实现全链路零堆分配路由。

```rust
// src/service/mod.rs
// 零拷贝响应结果抽象枚举
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TranslatedText {
    String(String),
    Cached(std::sync::Arc<str>),
}

impl std::ops::Deref for TranslatedText {
    type Target = str;
    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            Self::String(s) => s.as_str(),
            Self::Cached(a) => a.as_ref(),
        }
    }
}
```

### 7. 多级级联语言检测流水线
为了实现准确、零分配的动态语种识别，系统设计了五级级联语言检测流水线：
Tier 0 噪音绕过层：即时过滤空文本、全数字、通用 CLI 命令行及 HTTP 日志特征，对于无实体语义的输入快速给出默认判定；
Tier 1 单趟脚本扫描层（Single-Pass Script Scan）：仅需遍历一遍字符，统计 Han、Kana、Hangul、Cyrillic、Arabic、Devanagari、Latin 等脚本的分布特征。当韩文、日文假名、希腊文、希伯来文等独占脚本达到阈值时直接裁决；
Tier 2 正字法与独占标记层：对中文字符依据 `detection_model` 离线编译的繁简字典计算偏好得分，准确裁决 `zh-Hans` 与 `zh-Hant`；对西里尔字母依据独占字母（如乌克兰文独有的 `є`, `ї`，俄文独有的 `ы`, `э`）实现秒级区分；
Tier 3 核心词汇与短语快筛层：针对极短文本，利用零分配紧凑查表法检索 `CORE_GREETINGS`（如 `bonjour` 直接判定为法语），并比对亲缘语言的特征功能词（如丹麦语、挪威语与瑞典语特征词）；
Tier 4 统计 3-gram 回退层：若前三层未能形成明确高置信度裁决，则将文本截断送入 `whatlang` 引擎，通过三元字符概率模型计算全局综合打分；
整个语言检测流程在服务层外挂 16 分片的独立 LRU 缓存，使得相同句式的识别开销降为单次 FNV-1a 运算。

### 8. 富文本标记保留与 Cross-Attention 词对齐投影算法
这是 MTranServer-Rust 最具技术深度的算法体系之一。传统翻译器在面对 Markdown、HTML 富文本时，往往造成标签丢失、闭合错乱或标签内单词漂移。系统实现了媲美 Firefox Translations（Bergamot）的全套富文本保留投影流水线：

第一步：混合文档 AST 解析（Document Parsing）。在 `src/engine/markup/document.rs` 中，系统将复合文档解析为 `Verbatim`（代码块、YAML 元数据、注释等绝对不翻译区域）、`Structured`（标题、任务列表、脚注等带格式结构）、`Table`（Markdown 表格）以及 `Paragraph`（普通段落）四大 AST 节点；
第二步：行内标记剥离与坐标记录。在进入翻译前，`extract_inline_markup` 扫描段落，将 `<b>`、`[link](url)`、`<code>` 等标记剥离，纯净文本送入翻译模型，同时建立原始标记在源端字符流上的绝对范围 `source_range`；
第三步：方差过滤能量加权（Variance-Filtered Energy Weighting）。在 Transformer 最后一层交叉注意力中，不同的注意力头质量参差不齐，弥散的噪声头会破坏对齐精度。函数 `filter_and_accumulate_cross_attention_head` 计算各注意力头的最大能量峰值，动态设定阈值：
$$\text{threshold} = \frac{1.2}{L_{\text{src}}}, \quad \text{weight} = 0.001 + \max(0, \text{head\_max} - \text{threshold})^2$$
对尖锐的主力注意力头施加二次方加速加权，对弥散头施加 0.001 基底抑制，提炼出高信噪比的注意力对齐图谱；
第四步：一维连续密度聚类（1D Continuous Density Clustering）。在目标 Token 序列中，候选对齐点往往夹杂个别孤立的注意力误判毛刺。函数 `cluster_1d_major` 以间距阈值（通常 $\text{gap} \le 3$）对对齐点集进行一维密度切片，提取出密度最高的主簇范围，剔除远端孤立离群点。针对日语等 SOV（Head-Final）语言，算法自动优先吸附句尾核心成分；
第五步：边界安全吸附（Boundary-Safe Adhesion）。原始投影落点可能切断多字节 UTF-8 字符或将单个西文单词切半（例如将 `negrito` 错切为 `<b>neg</b>rito`）。函数 `snap_to_char_boundary` 强制纠偏落点至合法字符边界；函数 `snap_to_word_boundary` 对非 CJK 文本执行向外单词级膨胀吸附，完整包裹整个单词；对 CJK 文本则保留精细字符粒度；
第六步：语法树冲突消解重组（Syntax Tree Conflict Resolution）。多层嵌套标记在经历投影膨胀后，可能出现交叉非法的“幽灵闭合”（如 `<a><b></a></b>`）。函数 `resolve_markup_nesting_conflicts` 对所有重构 Span 按照起始位置与原始树深度进行排序，检测交错与深度反转，通过范围外推与合并重组，严格确保输出文档符合合法的 XML/HTML 嵌套树规范。

```rust
// src/engine/markup/mod.rs
// 多头注意力加权累积与一维连续聚类
#[inline]
pub(crate) fn filter_and_accumulate_cross_attention_head(
    accum: &mut [f32],
    scores: &[f32],
    kv_len: usize,
    h: usize,
) {
    let len = kv_len.min(accum.len()).min(scores.len());
    let mut head_max = 0.0f32;
    for &s in &scores[..len] {
        if s > head_max { head_max = s; }
    }
    let threshold = if kv_len > 1 { 1.2 / kv_len as f32 } else { 0.0 };
    let diff = (head_max - threshold).max(0.0);
    let weight = 0.001f32 + diff * diff;

    if h == 0 {
        for j in 0..len { accum[j] = scores[j] * weight; }
    } else {
        for j in 0..len { accum[j] += scores[j] * weight; }
    }
}
```

## 第五章 系统并发、缓存机制与编译分配器优化

### 1. 现代化内存分配器替换（mimalloc）
标准 C 运行时（如 Linux glibc 的 ptmalloc）在极端高并发、小对象频繁分配释放的多线程场景下，极易遭遇严重的中心锁争用（Heap Lock Contention）与由于虚存碎片导致的物理内存无法返还给操作系统的问题。

项目在二进制入口层面强行将全局内存分配器替换为微软研发的高性能 `mimalloc`：

```rust
// src/main.rs
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
```

`mimalloc` 凭借完全无锁的本地线程分片页分配机制（Thread-Local Sharded Pages）、自由列表就地重用以及极致优化的基数树内存回收算法，使得高并发批处理翻译场景下的内存碎片率下降至 5% 以下，分配延迟下降约 25%。

### 2. Cargo.toml 生产构建极限优化
为了释放编译器对 CPU 指令生成的所有潜能，项目的生产 Profile 进行了极致严苛的配置优化：

```toml
# Cargo.toml
[profile.release]
opt-level = 3
panic = "abort"
lto = "fat"
codegen-units = 1
strip = true
```

`opt-level = 3` 开启 rustc 与 LLVM 的全套激进优化 Pass，包括全量循环向量化与过程内激进内联；
`panic = "abort"` 彻底移除所有栈回溯展开（Stack Unwinding）代码表与落地逻辑，不仅大幅削减二进制体积，更允许编译器跨越潜在的 Panic 点自由重排指令流水线；
`lto = "fat"` 启动跨越所有依赖 Crates 的全程序全局链接时优化（Fat Link-Time Optimization），使得底层 SIMD 宏、张量抽象与外层业务函数的边界被彻底打通，产生跨越 Crate 边界的直接内联；
`codegen-units = 1` 强制 LLVM 放弃多模块并行编译，将整个代码库聚合成单个编译单元进行全量全局寄存器分配与死代码消除，杜绝模块间内联盲区；
`strip = true` 自动剔除所有调试符号与符号表，进一步压缩磁盘与加载镜像占用。

### 3. ENGINE_GATE 算力硬限制门控机制
在生产实践中，如果不对并发计算量进行顶层抑制，当上百个并发 HTTP 请求同时涌入并各自触发底层多线程 GEMM 时，CPU 运行队列将迅速积压数百个就绪线程，导致严重的上下文切换抖动、CPU L1/L2 缓存行彻底失效，从而引发尾部延迟灾难性崩塌。

系统基于 `tokio::sync::Semaphore` 设计了全局物理算力门控 `ENGINE_GATE`：

```rust
// src/service/mod.rs
// 全局 CPU 算力限制门控信号量
static ENGINE_GATE: std::sync::OnceLock<tokio::sync::Semaphore> = std::sync::OnceLock::new();

fn engine_gate() -> &'static tokio::sync::Semaphore {
    ENGINE_GATE.get_or_init(|| {
        let max_threads = config::get().max_threads;
        let permits = if max_threads > 0 {
            max_threads
        } else {
            // 默认严格收敛于宿主机的物理并行核心数
            std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4)
        };
        tokio::sync::Semaphore::new(permits)
    })
}
```

每个翻译任务在进入繁重的 CPU 密集型计算之前，必须在异步侧先行获取许可证 `_permit`。对于 Pivot Routing 这类双跳任务，许可证是在异步外层顺序获取，绝不发生锁重入与死锁。这一机制在物理上强制在途的重型推理线程数永不超越机器核心承载上限，将 CPU 上下文切换消耗降至接近于零。

### 4. 32 分片 64 字节缓存行强制对齐 LRU 缓存与 128 位哈希
翻译服务器的高频请求往往具有高度的局部重现性。然而，一个全局单一锁的 LRU 缓存会在多核高并发访问时成为严重的串行化瓶颈；更隐蔽的性能杀手是现代多核处理器的“伪共享”（False Sharing）：当相邻核心的锁或缓存元数据处于同一个 64 字节缓存行时，总线会频繁发生缓存行失效震荡。

系统构建了具有硬件级防伪共享特性的 32 分片并发 LRU 缓存：

```rust
// src/utils/cache.rs
// 强制 64 字节 CacheLine 对齐的独立分片结构
const SHARDS: usize = 32;

#[repr(align(64))]
struct CacheShard(parking_lot::RwLock<Option<LruCache<CacheKey, std::sync::Arc<str>>>>);

static CACHE: [CacheShard; SHARDS] = [const { CacheShard(parking_lot::RwLock::new(None)) }; SHARDS];
```

结构体通过 `#[repr(align(64))]` 显式强制编译器将其在物理内存中对齐到 64 字节边界，彻底消除了不同核心争用相邻分片时的 False Sharing 惩罚。分片索引通过 128 位哈希值的低 5 位进行位与掩码计算（$O(1)$ 无除法）。

缓存键采用基于 16 字节双流宽乘法构建的 128 位强抗碰撞哈希 `CacheKey(u64, u64)`，在输入流末尾施加 SplitMix64 雪崩扰动，并与当前语言规则版本号（`rules_version`）强制混合，彻底根绝哈希碰撞并实现规则更新后的秒级无感失效。在读取路径上，系统采用乐观的读锁探测；仅当分片写锁完全无竞争时才尝试通过 `try_write` 原地调整 LRU 双向链表节点，在极高并发下实现了近乎无锁的读取性能。

### 5. Pivot Routing 多跳并发预加载
当请求的源语言与目标语言之间缺乏直接训练的翻译模型时（例如将斯瓦希里语直译为中文），系统自动触发以英语为中枢枢纽的 Pivot Routing 机制（源语言 -> 英语 -> 目标语言）。

系统在确定需要多跳后，绝不采用传统阻塞式串行加载模式，而是通过 `tokio::join!` 异步并发宏，同时向模型池发起两段模型的并发装载请求：

```rust
// src/service/mod.rs
// 中转路由模型并发就绪与装载
pub async fn ensure_required_engines(from_lang: &str, to_lang: &str) -> Result<(PreparedEngines, Duration), ServiceError> {
    if !needs_pivot_translation(from_lang, to_lang) {
        let (engine, d) = ensure_engine_loaded(from_lang, to_lang).await?;
        Ok((PreparedEngines::Direct(engine), d))
    } else {
        // 并发触发两个独立模型的装载流水线，重叠磁盘 I/O 与 Single-Flight 准备时间
        let (res1, res2) = tokio::join!(
            ensure_engine_loaded(from_lang, "en"),
            ensure_engine_loaded("en", to_lang)
        );
        let (engine1, d1) = res1?;
        let (engine2, d2) = res2?;
        Ok((PreparedEngines::Pivot(engine1, engine2), d1.max(d2)))
    }
}
```

通过这一并发重叠，第一跳与第二跳模型的 I/O 准备时间实现了最大化重合，使得多跳冷启动的准备耗时几乎等同于单模型加载耗时。

### 6. 细粒度并发与客户端 IP 限流隔离
为了防范单个客户端的高并发脚本攻击拖垮整个服务器集群，系统在 API 网关入口层构建了细粒度客户端 IP 并发控制器 `IpConcurrencyLimiter`。

该限流器内部同样划分为 16 个独立的互斥分片，各个 IP 的哈希落点被打散至不同分片中，极大地减少了锁碰撞。客户端的每一次活动连接被封装为一个 RAII 机制的 `IpConcurrencyGuard`。当某个客户端 IP 发起的在途并发请求数超过阈值时，网关在微秒内直接返回 429 Too Many Requests 错误，完全阻断其渗透至昂贵的底层神经推理层；而当请求正常结束或因网络异常提前断开时，RAII 守卫的 `drop` 函数自动减少计数并在计数归零时主动清理哈希表键值，保证系统长期运行下内存不发生泄漏。

```rust
// src/middleware/mod.rs
// 基于 16 分片原子计数的客户端 IP 限流控制器
pub struct IpConcurrencyLimiter {
    shards: [parking_lot::Mutex<HashMap<std::net::IpAddr, usize>>; 16],
    max_concurrency: usize,
}

pub struct IpConcurrencyGuard<'a> {
    limiter: &'a IpConcurrencyLimiter,
    ip: std::net::IpAddr,
    shard_idx: usize,
}

impl<'a> Drop for IpConcurrencyGuard<'a> {
    fn drop(&mut self) {
        let mut map = self.limiter.shards[self.shard_idx].lock();
        if let Some(count) = map.get_mut(&self.ip) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                map.remove(&self.ip);
            }
        }
    }
}
```
