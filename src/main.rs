mod net;

use nalgebra::DMatrix;

use crate::net::Net;
use net::cost_funcs::LossFunction;
use net::act_funcs::ActivationFunction;


pub struct Hyperparams {
    epochs: usize,
    learning_rate: f64,
}

impl Hyperparams {
    pub fn new() -> Self {
        Hyperparams {
            epochs: 100000,
            learning_rate: 0.5,
        }
    }
}


fn main() {
    let data = DMatrix::from_row_slice(2, 4, &[
        0.0, 1.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 1.0,
    ]);

    let label = DMatrix::from_row_slice(1, 4, &[
        0.0, 1.0, 1.0, 0.0,
    ]);

    let arch = vec![2, 4, 1];

    let params = Hyperparams::new();
    let loss_func = LossFunction::SquaredError;
    let act_func = ActivationFunction::Sigmoid;

    let mut net = Net::new(arch, loss_func);
    net.train(&data, &label, params, &act_func);

    for i in 0..4 {
        let col_input: Vec<f64> = data.column(i).iter().cloned().collect();
        let x = DMatrix::from_vec(data.nrows(), 1, col_input);

        let col_label: Vec<f64> = label.column(i).iter().cloned().collect();
        let y = DMatrix::from_vec(label.nrows(), 1, col_label);

        let out = net.predict(&x, &act_func);

        println!("{}, {} -> {}: {}", x[(0, 0)], x[(1, 0)], y[(0, 0)], out[(0, 0)]);
    }
}
