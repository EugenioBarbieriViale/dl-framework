use crate::nn::matrix::Mat;
use std::f64::consts::E;

#[derive(Copy, Clone)]
pub enum Function {
    SIGMOID,
    RELU,
    LEAKY_RELU(f64),
    TANH,
    SOFTMAX,
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(x))
}

fn relu(x: f64) -> f64 {
    match x > 0f64 {
        true => x,
        false => 0f64,
    }
}

fn leaky_relu(x: f64, neg_slope: f64) -> f64 {
    match x >= 0f64 {
        true => x,
        false => x * neg_slope,
    }
}

fn tanh_f(x: f64) -> f64 {
    x.tanh()
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
                Function::RELU => elems[i][j] = relu(mat.elems[i][j]),
                Function::LEAKY_RELU(neg_slope) => elems[i][j] = leaky_relu(mat.elems[i][j], neg_slope),
                Function::TANH => elems[i][j] = tanh_f(mat.elems[i][j]),
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
