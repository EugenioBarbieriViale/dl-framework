use nalgebra::DMatrix;
// use nalgebra::DVector;
use std::f64::consts::E;


fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn softmax(x: &DMatrix<f64>) -> DMatrix<f64> {
    let e = x.map(|xi| xi.exp());
    let s = e.sum();
    e.map(|xi| xi / s)
}

#[derive(Clone)]
pub enum ActivationFunction {
    Sigmoid,
    ReLU,
    Softmax,
}

impl ActivationFunction {
    pub fn compute(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        match self {
            ActivationFunction::Sigmoid => x.map(|x| sigmoid(x)),
            ActivationFunction::ReLU => x.map(|x| if x > 0.0 {x} else {0.0}),
            ActivationFunction::Softmax => {
                if x.ncols() != 1 { panic!("Cannot softmax a non vector") };
                softmax(x)
            }
        }
    }

    pub fn derivative(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        match self {
            ActivationFunction::Sigmoid => x.map(|x| sigmoid(x) * (1.0 - sigmoid(x))),
            ActivationFunction::ReLU => x.map(|x| if x > 0.0 {1.0} else {0.0}),
            ActivationFunction::Softmax => panic!("Use softmax and cross-entropy combined"),
        }
    }
}


pub enum LossFunction {
    SquaredError,
    CrossEntropy,
}

impl LossFunction {
    pub fn compute(&self, y_pred: &DMatrix<f64>, y_true: &DMatrix<f64>) -> f64 {
        match self {
            LossFunction::SquaredError => (y_pred - y_true).map(|x| x * x).sum() / y_pred.len() as f64,
            LossFunction::CrossEntropy => -(y_pred.map(|x| x.log(E)).component_mul(y_true)).sum(),
        }
    }

    pub fn gradient(&self, y_pred: &DMatrix<f64>, y_true: &DMatrix<f64>) -> DMatrix<f64> {
        match self {
            LossFunction::SquaredError => (y_pred - y_true) * 2.0,
            LossFunction::CrossEntropy => y_pred - y_true,
        }
    }
}
