//! Neural network layer trait and dense layer implementation.

use ndarray::Array2;

/// A neural network layer.
pub trait Layer {
    /// Forward pass: input -> output.
    fn forward(&mut self, input: &Array2<f64>) -> Array2<f64>;
    /// Backward pass: output gradient -> input gradient. Updates internal weights.
    fn backward(&mut self, grad_output: &Array2<f64>, learning_rate: f64) -> Array2<f64>;
}

/// Dense (fully-connected) layer.
pub struct Dense {
    pub weights: Array2<f64>,
    pub bias: ndarray::Array1<f64>,
    input_cache: Option<Array2<f64>>,
}

impl Dense {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        use ndarray_rand::RandomExt;
        use rand_distr::Normal;
        // Xavier initialization
        let std = (2.0 / (input_size + output_size) as f64).sqrt();
        Self {
            weights: Array2::random((input_size, output_size), Normal::new(0.0, std).unwrap()),
            bias: ndarray::Array1::zeros(output_size),
            input_cache: None,
        }
    }
}

impl Layer for Dense {
    fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
        self.input_cache = Some(input.clone());
        input.dot(&self.weights) + &self.bias
    }

    fn backward(&mut self, grad_output: &Array2<f64>, learning_rate: f64) -> Array2<f64> {
        let input = self.input_cache.as_ref().expect("forward() not called");
        let n = input.nrows() as f64;

        let grad_weights = input.t().dot(grad_output) / n;
        let grad_bias = grad_output.mean_axis(ndarray::Axis(0)).unwrap();
        let grad_input = grad_output.dot(&self.weights.t());

        self.weights = &self.weights - &(learning_rate * &grad_weights);
        self.bias = &self.bias - &(learning_rate * &grad_bias);

        grad_input
    }
}
