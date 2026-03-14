pub mod octonion;
pub mod sedenion;
pub mod cayley_dickson;
pub mod bsp;

pub use octonion::Octonion;
pub use sedenion::Sedenion;
pub use cayley_dickson::{double_multiply, double_conjugate, double_norm};
pub use bsp::BspNode;
