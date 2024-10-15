use crate::nn::matrix::Mat;
use std::f64::consts::E;

pub enum Function {
    SIGMOID,
    RELU,
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

fn tanh_f(x: f64) -> f64 {
    x.tanh()
}

fn softmax(vec: &Vec<f64>) -> Vec<f64> {
    vec.iter().map(|x| E.powf(*x) / vec.iter().sum::<f64>()).collect()
}

pub fn activate(mat: &Mat, func: Function) -> Mat {
    let mut elems = mat.elems.clone();

    'outer: for i in 0..mat.rows {
        for j in 0..mat.cols {
            match func {
                Function::SIGMOID => elems[i][j] = sigmoid(mat.elems[i][j]),
                Function::RELU => elems[i][j] = relu(mat.elems[i][j]),
                Function::TANH => elems[i][j] = tanh_f(mat.elems[i][j]),
                Function::SOFTMAX => {
                    match mat.rows == 1 {
                        false => break 'outer,
                        true => elems[0] = softmax(&elems[0]),
                    }
                }
            }
        }
    }

    Mat {
        rows: mat.rows,
        cols: mat.cols,
        elems,
    }
}
