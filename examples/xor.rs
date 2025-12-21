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
            epochs: 10000,
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

    for e in 0..params.epochs {
        for i in 0..4 {
            let x = get_col(i, &data);
            let y = get_col(i, &label);

            net.train(&x, &y, &params, &act_func);
        }

        if e % 100 == 0 {
            println!("Epoch: {} cost: {}", e, net.cost);
        }
    }

    for i in 0..4 {
        let x = get_col(i, &data);
        let y = get_col(i, &label);

        let out = net.predict(&x, &act_func);

        println!("{}, {} -> {}: {}", x[(0, 0)], x[(1, 0)], y[(0, 0)], out[(0, 0)]);
    }
}

fn get_col(i: usize, m: &DMatrix<f64>) -> DMatrix<f64> {
    DMatrix::from_vec(m.nrows(), 1, m.column(i).iter().cloned().collect())
}
