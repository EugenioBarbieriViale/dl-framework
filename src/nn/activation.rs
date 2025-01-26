use crate::nn::matrix::Mat;
use std::f64::consts::E;

#[derive(Copy, Clone)]
pub enum Function {
    SIGMOID,
    D_SIGMOID,

    RELU(f64),
    D_RELU(f64),

    TANH,
    D_TANH,

    SOFTMAX,
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(x))
}

fn d_sigmoid(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

fn relu(x: f64, neg_slope: f64) -> f64 {
    match x > 0f64 {
        true => x,
        false => x * neg_slope,
    }
}

fn d_relu(x: f64, neg_slope: f64) -> f64 {
    match x > 0f64 {
        true => 1f64,
        false => neg_slope,
    }
}

fn tanh_f(x: f64) -> f64 {
    x.tanh()
}

fn d_tanh_f(x: f64) -> f64 {
    (1.0 / x.cosh()) * (1.0 / x.cosh())
}

fn softmax(vec: &Vec<f64>) -> Vec<f64> {
    vec.iter().map(|x| E.powf(*x) / vec.iter().sum::<f64>()).collect()
}

pub fn activate(mat: &Mat, func: Function) -> Mat {
    let mut elems = mat.elems.clone();

    for i in 0..mat.rows {
        for j in 0..mat.cols {
            match func {
                Function::SIGMOID => elems[i][j] = sigmoid(mat.elems[i][j]),
                Function::D_SIGMOID => elems[i][j] = d_sigmoid(mat.elems[i][j]),

                Function::RELU(neg_slope) => elems[i][j] = relu(mat.elems[i][j], neg_slope),
                Function::D_RELU(neg_slope) => elems[i][j] = d_relu(mat.elems[i][j], neg_slope),

                Function::TANH => elems[i][j] = tanh_f(mat.elems[i][j]),
                Function::D_TANH => elems[i][j] = d_tanh_f(mat.elems[i][j]),

                Function::SOFTMAX => todo!(),
            }
        }
    }

    Mat {
        rows: mat.rows,
        cols: mat.cols,
        elems,
    }
}
