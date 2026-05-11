use nalgebra::DMatrix;
use rand_distr::{Distribution, Normal};

pub enum Fan {
    In,
    Out,
}

pub enum Initialization {
    Zero,
    Random,
    Kaiming(Fan),
}

pub fn new(i: &Initialization, dims: (usize, usize)) -> DMatrix<f64> {
    match i {
        Initialization::Zero => DMatrix::<f64>::zeros(dims.0, dims.1),
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
