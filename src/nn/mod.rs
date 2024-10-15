pub mod matrix;
mod activation;

use matrix::Mat;
use activation::*;

macro_rules! square {
    ($s:expr) => {
        ($s)*($s)
    }
}


pub fn forward(input: &Mat, l1: &Mat, l2: &Mat) -> Mat {
    let hidden = activate(&input.dot(l1), activation::Function::SIGMOID);
    activate(&hidden.dot(l2), activation::Function::SIGMOID)
}

pub fn loss(output: &Mat, label: &Mat) -> Option<f64> {
   match label.cols == output.cols && output.rows == 1 && label.rows == 1 {
       false => None,
       true => {
           let mut cost = 0.0;
           for i in 0..output.cols {
               cost += square!(output.elems[0][i] - label.elems[0][i]);
           }
           cost /= output.cols as f64;
           Some(cost)
       },
   }
}
