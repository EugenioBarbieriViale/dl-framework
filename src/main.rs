mod nn;

use crate::nn::{*, matrix::Mat};

const EPOCHS: u32 = 10000;
const RATE: f64 = 2.0;

fn main() {
    let data: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let labels: Vec<f64> = vec![
        0.0,
        1.0,
        1.0,
        1.0,
    ];

    let mut l1 = Mat::new_rand(2, 2);
    let mut l2 = Mat::new_rand(2, 1);

    let mut out = Mat::new(2, 1);
    let mut label = Mat::new(1, 1);

    for i in 0..EPOCHS {
        for j in 0..labels.len() {
            let input = Mat::from_vec(1, 2, data[j].clone());
            label = Mat::from_vec(1, 1, labels[j..j+1].to_vec());

            out = forward(&input, &l1, &l2);

            let mut g1 = finite_diff1(&input, &out, &label, &mut l1, &l2);
            let mut g2 = finite_diff2(&input, &out, &label, &l1, &mut l2);

            g1.scalar_mult(RATE);
            g2.scalar_mult(RATE);

            l1.sub(&g1);
            l2.sub(&g2);
        }
        let c = loss(&out, &label);
        println!("{}", c);
    }

    for i in 0..4 {
        let input = Mat::from_vec(1, 2, data[i].clone());
        let outf = forward(&input, &l1, &l2);
        println!("{:?}: {:?} -> {:?}", &data[i], &labels[i], outf.elems);
    }
}
