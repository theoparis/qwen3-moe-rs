mod algorithm;
mod precision;
mod tiling_scheme;

#[macro_export]
macro_rules! testgen_convolution_accelerated {
    () => {
        mod conv2d_accelerated {
            use super::*;
            use cubecl::std::CubeOption;
            use cubek_matmul::components::tile::io::Strided;
            type TMM = cubek_matmul::components::tile::cmma::CmmaMatmul<CubeOption<Strided>>;

            #[cfg(all(feature = "conv_tests_plane", not(feature = "conv_tests_mma")))]
            $crate::testgen_convolution_accelerated_algorithm!();

            #[cfg(all(feature = "conv_tests_plane", feature = "conv_tests_mma"))]
            mod cmma {
                use super::*;
                type TMM = cubek_matmul::components::tile::cmma::CmmaMatmul<CubeOption<Strided>>;

                $crate::testgen_convolution_accelerated_algorithm!();
            }

            #[cfg(all(feature = "conv_tests_plane", feature = "conv_tests_mma"))]
            mod mma {
                use super::*;
                type TMM = cubek_matmul::components::tile::mma::MmaMatmul<
                    Strided,
                    Strided,
                    CubeOption<Strided>,
                >;

                $crate::testgen_convolution_accelerated_algorithm!();
            }
        }
    };
}
