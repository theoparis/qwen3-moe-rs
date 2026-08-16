//! Shared CubeCL helpers for custom kernels that run on CUDA and wgpu/Metal/Vulkan.

use burn::tensor::DType;
use burn_cubecl::CubeRuntime;
use burn_cubecl::tensor::CubeTensor;

pub fn alloc_f32<R: CubeRuntime>(like: &CubeTensor<R>, shape: &[usize]) -> CubeTensor<R> {
    let n: usize = shape.iter().product();
    let buffer = like.client.empty(n * DType::F32.size());
    CubeTensor::new_contiguous(
        like.client.clone(),
        like.device.clone(),
        shape.to_vec().into(),
        buffer,
        DType::F32,
    )
}
