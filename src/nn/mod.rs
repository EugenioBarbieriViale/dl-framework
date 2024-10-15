pub mod matrix;
mod activation;

use matrix::Mat;
use activation::{Function, activate};

macro_rules! square {
    ($s:expr) => {
        ($s)*($s)
    }
}


pub fn forward(input: &Mat, l1: &Mat, l2: &Mat) -> Mat {
    let hidden = activate(&input.dot(l1), Function::SIGMOID);
    activate(&hidden.dot(l2), Function::SIGMOID)
}

pub fn loss(output: &Mat, label: &Mat) -> f64 {
   // match label.cols == output.cols && output.rows == 1 && label.rows == 1 {
   //     false => None,
   //     true => {
   //         let mut cost = 0.0;
   //         for i in 0..output.cols {
   //             cost += square!(output.elems[0][i] - label.elems[0][i]);
   //         }
   //         cost /= output.cols as f64;
   //         Some(cost)
   //     },
   // }

   let mut cost = 0.0;
   for i in 0..output.cols {
       cost += square!(output.elems[0][i] - label.elems[0][i]);
   }
   cost /= output.cols as f64;
   cost
}

pub fn finite_diff1(input: &Mat, out: &Mat, label: &Mat, l1: &mut Mat, l2: &Mat) -> Mat {
    let out = forward(&input, &l1, &l2);
    let prev_loss = loss(&out, &label);

    let mut g1 = Mat::new(l1.rows, l1.cols);

    for i in 0..l1.rows {
        for j in 0..l1.cols {
            l1.elems[i][j] += 1e-2;
            g1.elems[i][j] = loss(&forward(&input, &l1, &l2), &label);
            l1.elems[i][j] -= 1e-2;
        }
    }
    g1
}

pub fn finite_diff2(input: &Mat, out: &Mat, label: &Mat, l1: &Mat, l2: &mut Mat) -> Mat {
    let out = forward(&input, &l1, &l2);
    let prev_loss = loss(&out, &label);

    let mut g2 = Mat::new(l2.rows, l2.cols);

    for i in 0..l2.rows {
        for j in 0..l2.cols {
            l2.elems[i][j] += 1e-2;
            g2.elems[i][j] = loss(&forward(&input, &l1, &l2), &label);
            l2.elems[i][j] -= 1e-2;
        }
    }
    g2
}
