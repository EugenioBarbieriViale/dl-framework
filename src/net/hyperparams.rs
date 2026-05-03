use nalgebra::DMatrix;
use rand_distr::{Distribution, Normal};

pub struct Hyperparams {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
}

impl Hyperparams {
    pub fn new(epochs: usize, bs: usize, lr: f64) -> Self {
        Hyperparams {
            epochs,
            batch_size: bs,
            learning_rate: lr,
        }
    }
}

pub enum Fan {
    In,
    Out,
}

pub enum Initialization {
    Random,
    Kaiming(Fan),
}

pub fn new(i: &Initialization, dims: (usize, usize)) -> DMatrix<f64> {
    match i {
        Initialization::Random => DMatrix::<f64>::new_random(dims.0, dims.1),
        Initialization::Kaiming(fan) => kaiming(dims, fan),
    }
}

fn kaiming(dims: (usize, usize), fan: &Fan) -> DMatrix<f64> {
    let n = match fan {
        Fan::In => dims.0,
        Fan::Out => dims.1,
    };
    let normal = Normal::new(0.0, (2.0 / n as f64).sqrt()).unwrap();

    let mut rng = rand::rng();
    let v: Vec<f64> = (0..dims.0 * dims.1)
        .map(|_| normal.sample(&mut rng))
        .collect();

    DMatrix::<f64>::from_row_slice(dims.0, dims.1, &v)
}
