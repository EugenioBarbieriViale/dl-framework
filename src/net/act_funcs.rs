use nalgebra::DMatrix;

pub enum ActivationFunction {
    Sigmoid,
    ReLU,
    Softmax,
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

impl ActivationFunction {
    pub fn compute(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        match self {
            ActivationFunction::Sigmoid => x.map(|x| sigmoid(x)),
            ActivationFunction::ReLU => x.map(|x| if x > 0.0 {x} else {0.0}),
            ActivationFunction::Softmax => x.map(|xi| xi.exp() / x.exp().sum()),
        }
    }

    pub fn derivative(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        match self {
            ActivationFunction::Sigmoid => x.map(|x| sigmoid(x) * (1.0 - sigmoid(x))),
            ActivationFunction::ReLU => x.map(|x| if x > 0.0 {1.0} else {0.0}),
            ActivationFunction::Softmax => todo!(),
        }
    }
}
