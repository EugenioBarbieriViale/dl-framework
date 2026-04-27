pub struct Hyperparams {
    pub epochs: usize,
    pub learning_rate: f64,
}

impl Hyperparams {
    pub fn new(epochs: usize, lr: f64) -> Self {
        Hyperparams {
            epochs,
            learning_rate: lr,
        }
    }
}
