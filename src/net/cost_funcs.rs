use nalgebra::DMatrix;

pub enum LossFunction {
    SquaredError,
    CrossEntropy,
}

impl LossFunction {
    pub fn compute(&self, y_pred: &DMatrix<f64>, y_true: &DMatrix<f64>) -> f64 {
        match self {
            LossFunction::SquaredError => (y_pred - y_true).map(|x| x * x).sum() / y_pred.len() as f64,
            LossFunction::CrossEntropy => todo!(),
            // LossFunction::CrossEntropy => y_pred.cross_entropy(&y_true).sum(),
        }
    }

    pub fn gradient(&self, y_pred: &DMatrix<f64>, y_true: &DMatrix<f64>) -> DMatrix<f64> {
        match self {
            LossFunction::SquaredError => (y_pred - y_true) * 2.0,
            LossFunction::CrossEntropy => todo!(),
            // LossFunction::CrossEntropy => y_pred.cross_entropy(&y_true).sum(),
        }
    }
}
