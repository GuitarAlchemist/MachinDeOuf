//! GPU-accelerated compute via WGPU.
//!
//! Cross-platform: Vulkan/DX12 on Windows (NVIDIA), Metal on macOS.
//! Provides cosine similarity, dot product, matrix multiply, and batch vector search.

pub mod context;
pub mod similarity;
pub mod matmul;
pub mod batch;
