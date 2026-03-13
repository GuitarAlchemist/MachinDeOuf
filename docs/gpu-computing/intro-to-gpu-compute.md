# Intro to GPU Compute

## The Problem

You have a batch of 10,000 embedding vectors, each with 768 dimensions, and you need to find the cosine similarity between every pair. On a CPU, that is ~50 million dot products, each requiring 768 multiply-adds. Even on a fast machine, this takes seconds. A GPU has thousands of cores that can each compute a dot product simultaneously --- turning seconds into milliseconds. But GPU programming has a reputation for being painful: vendor-specific APIs, manual memory management, shader compilation, synchronization. You need a way to get GPU acceleration without drowning in boilerplate.

---

## The Intuition

Think of a CPU as a single brilliant mathematician who works through problems one at a time, very fast. A GPU is a stadium full of 5,000 average mathematicians who can each solve a simple problem simultaneously. If your workload is one giant equation with deep dependencies between steps, the single brilliant mathematician wins. But if your workload is thousands of *independent* problems --- like computing 10,000 dot products --- the stadium full of workers finishes in the time it takes to do a single one.

The cost of using the stadium: you have to write your problem on a whiteboard in a language the workers understand (a **shader**), bus the data to the stadium (upload to GPU memory), and bus the results back (readback). For small problems, the bus ride takes longer than the computation. For large problems, it is overwhelmingly worth it.

---

## How It Works

### The WGPU stack

MachinDeOuf uses [WGPU](https://wgpu.rs/), a cross-platform GPU abstraction layer:

| Platform | Backend |
|----------|---------|
| Windows | Vulkan or DX12 (auto-selected) |
| macOS | Metal |
| Linux | Vulkan |

You write compute shaders in **WGSL** (WebGPU Shading Language), a simple C-like language. WGPU compiles them at runtime for whatever backend is available.

### The compute pipeline

Every GPU computation follows this pattern:

```
1. Initialize  -> GpuContext::new()
2. Upload      -> create_buffer_init(label, &data)
3. Compile     -> create_compute_pipeline(label, shader_source, entry_point)
4. Bind        -> bind buffers to shader variables
5. Dispatch    -> launch N workgroups of M threads each
6. Readback    -> read_buffer(&output_buffer, size)
```

**In plain English:**

- **Initialize** finds a GPU and opens a connection (device + queue).
- **Upload** copies your f32 data from CPU RAM into GPU VRAM.
- **Compile** turns your WGSL shader source into native GPU instructions.
- **Bind** tells the shader "buffer 0 is your input A, buffer 1 is input B, buffer 2 is output."
- **Dispatch** says "launch this many thread groups; each group runs the shader on its slice of data."
- **Readback** copies results back to CPU RAM so you can use them in normal Rust code.

### Why f32, not f64?

GPUs are optimized for 32-bit floating point. Most consumer GPUs have **32x** more f32 throughput than f64. MachinDeOuf's CPU algorithms use `f64` for precision, but all GPU paths use `f32`. For the vast majority of ML workloads (similarity search, matrix multiply, neural net inference), f32 is more than sufficient.

---

## In Rust

> Full runnable example: [`examples/gpu/similarity_search.rs`](../../examples/gpu/similarity_search.rs)

### Initializing the GPU context

```rust
use machin_gpu::context::GpuContext;

// Synchronous initialization (blocks until GPU is ready)
let ctx = GpuContext::new().expect("No compatible GPU found");

println!("GPU: {}", ctx.gpu_name());     // e.g., "NVIDIA GeForce RTX 4090"
println!("Backend: {:?}", ctx.backend()); // e.g., Vulkan
```

For async code (inside a Tokio runtime):

```rust
use machin_gpu::context::GpuContext;

let ctx = GpuContext::new_async().await.expect("No compatible GPU found");
```

### Creating buffers

```rust
use machin_gpu::context::GpuContext;

let ctx = GpuContext::new().unwrap();

// Upload f32 data to GPU memory
let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
let gpu_buffer = ctx.create_buffer_init("my_data", &data);

// Create an empty output buffer (4 floats = 16 bytes)
let output = ctx.create_output_buffer("result", 16);

// Read results back after a computation
let results: Vec<f32> = ctx.read_buffer(&output, 16);
```

### Understanding the GpuContext fields

```rust
use machin_gpu::context::GpuContext;

let ctx = GpuContext::new().unwrap();

// Direct access to the underlying wgpu device and queue for advanced usage
let _device: &wgpu::Device = &ctx.device;
let _queue: &wgpu::Queue = &ctx.queue;

// Adapter info for diagnostics
let name = ctx.gpu_name();      // Human-readable GPU name
let backend = ctx.backend();     // Vulkan, DX12, Metal, etc.
```

### Handling "no GPU available"

```rust
use machin_gpu::context::GpuContext;

match GpuContext::new() {
    Ok(ctx) => {
        println!("Running on GPU: {}", ctx.gpu_name());
        // ... GPU-accelerated path ...
    }
    Err(e) => {
        eprintln!("GPU unavailable ({}), falling back to CPU", e);
        // ... CPU fallback path ...
    }
}
```

All `machin-gpu` modules provide CPU fallback functions alongside their GPU versions, so your code can degrade gracefully.

---

## When To Use This

| Workload | GPU worth it? | Why |
|----------|:------------:|-----|
| Similarity search over 10,000+ vectors | Yes | Thousands of independent dot products |
| Matrix multiply (large matrices) | Yes | Massively parallel multiply-accumulate |
| Single dot product of 2 vectors | No | Transfer overhead dominates |
| Sequential algorithm with data dependencies | No | GPU cannot parallelize serial work |
| Small dataset (< 1,000 items) | Rarely | CPU is fast enough; GPU setup is overhead |
| Batch inference over many inputs | Yes | Each input is independent |

**Rule of thumb:** if you can express your problem as "do the same thing to thousands of independent data points," GPU acceleration will help. The crossover point is typically around 1,000--10,000 items, depending on vector dimensionality.

---

## Key Parameters

| Parameter | What it controls | Guidance |
|-----------|-----------------|----------|
| Workgroup size | Threads per workgroup (set in WGSL: `@workgroup_size(256)`) | 256 is a safe default for most GPUs. Must be a power of 2. |
| Number of workgroups | Total parallelism: `ceil(data_size / workgroup_size)` | Computed automatically by MachinDeOuf functions |
| Buffer usage flags | `STORAGE`, `COPY_SRC`, `MAP_READ`, etc. | Handled by `create_buffer_init`, `create_output_buffer`, `create_readback_buffer` |
| Power preference | `HighPerformance` vs `LowPower` | MachinDeOuf defaults to `HighPerformance` (discrete GPU preferred) |

---

## Pitfalls

1. **GPU initialization is slow.** `GpuContext::new()` takes 10--100ms to enumerate adapters, request a device, and compile shaders. Create it once and reuse it across computations.

2. **Transfer overhead is real.** Uploading data to VRAM and reading results back takes time proportional to the data size. For small inputs (a few hundred floats), the transfer alone may exceed the CPU computation time.

3. **f32 precision loss.** If your algorithm is numerically sensitive (e.g., computing tiny differences between large numbers), the drop from f64 to f32 can introduce meaningful error. The CPU fallback functions use f32 too (for API compatibility), so compare against `machin-math` f64 routines for a ground truth.

4. **Not all GPUs are equal.** Integrated GPUs (Intel UHD, AMD APUs) have far less compute throughput than discrete GPUs (NVIDIA RTX, AMD RX). Test on your target hardware.

5. **Shader compilation is at runtime.** WGSL shaders are compiled when you first call a GPU function. Subsequent calls reuse the compiled pipeline. This means the first call is slower than expected.

---

## Going Further

- **[Similarity Search](./similarity-search.md):** GPU-accelerated cosine similarity, dot product, and batch top-k vector search.
- **[Matrix Multiply](./matrix-multiply-gpu.md):** GPU matrix multiplication for batch inference and neural network forward passes.
- WGSL specification: [https://www.w3.org/TR/WGSL/](https://www.w3.org/TR/WGSL/)
- WGPU documentation: [https://docs.rs/wgpu](https://docs.rs/wgpu)
- For understanding GPU architecture: "A Trip Through the Graphics Pipeline" gives excellent intuition about how GPUs process thousands of threads simultaneously.
