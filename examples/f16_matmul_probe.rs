use burn::backend::DType;
use burn::prelude::*;

fn main() {
    let device = Device::metal(Default::default());
    let a = Tensor::<2>::from_data([[1.0f32, 2.0], [3.0, 4.0]], &device).cast(DType::F16);
    let b = Tensor::<2>::from_data([[5.0f32, 6.0], [7.0, 8.0]], &device).cast(DType::F16);
    let c = a.matmul(b).cast(DType::F32);
    device.sync().ok();
    let data: Vec<f32> = c.into_data().to_vec().unwrap();
    println!("F16 matmul result: {:?} (expected [19,22,43,50])", data);
}
