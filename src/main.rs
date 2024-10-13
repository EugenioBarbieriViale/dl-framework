mod nn;
use crate::nn::*;

fn main() {
    let data: Vec<f64> = vec![
        0.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        1.0, 0.0, 0.0,
        1.0, 1.0, 1.0,
    ];

    let input = Mat::from(4, 3, data);

    let l1 = Mat::init(3, 4);
    let l2 = Mat::init(4, 3);

    let res = &l1.dot(&l2);
    res.print();
}
