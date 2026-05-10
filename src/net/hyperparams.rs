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
