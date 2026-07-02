pub mod global;
pub mod stage;

mod config;
mod error;
mod problem;
mod selection;

pub use config::*;
use cubecl::std::CubeOption;
use cubek_matmul::components::tile::{cmma::CmmaMatmul, io::Strided};
pub use error::*;
pub use problem::*;
pub use selection::*;

/// Convolution using `AcceleratedMatmul`
pub type AcceleratedConv = CmmaMatmul<CubeOption<Strided>>;
