mod convolution_test_launcher;
pub mod test_macros;
mod test_utils;

mod accelerated {
    crate::testgen_convolution_accelerated!();
}
