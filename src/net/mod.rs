pub mod cost_funcs;
pub mod act_funcs;

use nalgebra::DMatrix;
use cost_funcs::LossFunction;
use act_funcs::ActivationFunction;
use super::Hyperparams;


pub struct Net {
    layers: usize,
    arch: Vec<usize>,
    loss_function: LossFunction,
    weights: Vec<DMatrix<f64>>,
    biases: Vec<DMatrix<f64>>,
    zs: Vec<DMatrix<f64>>,
    activations: Vec<DMatrix<f64>>,
    cost: f64,
}

impl Net {
    pub fn new(arch: Vec<usize>, loss_function: LossFunction) -> Self {
        let layers = arch.len() - 1;

        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for i in 0..layers {
            let cols = arch[i];
            let rows = arch[i + 1];

            let weight_matrix = DMatrix::<f64>::new_random(rows, cols);
            let bias_matrix = DMatrix::<f64>::new_random(rows, 1);

            weights.push(weight_matrix);
            biases.push(bias_matrix);
        }

        let mut zs = Vec::with_capacity(layers);
        let mut activations = Vec::with_capacity(layers + 1);
        
        for i in 0..=layers {
            let size = arch[i];
            activations.push(DMatrix::<f64>::zeros(size, 1));
            if i < layers {
                zs.push(DMatrix::<f64>::zeros(arch[i + 1], 1));
            }
        }

        Net {
            layers,
            arch,
            loss_function,
            weights,
            biases,
            zs,
            activations,
            cost: 0.0,
        }
    }

    fn forward(&mut self, x: &DMatrix<f64>, act_func: &ActivationFunction) -> DMatrix<f64> {
        self.activations[0] = x.clone();

        for i in 0..self.layers {
            // println!("{}", self.activations[i]);
            // println!("{}", self.weights[i]);

            // self.zs[i] = (&self.weights[i] * &self.activations[i]) + &self.biases[i];
            // self.activations[i+1] = act_func.compute(&self.zs[i]);
            
            let weighted_sum = &self.weights[i] * &self.activations[i];
            let num_samples = weighted_sum.ncols();
            
            // Broadcast bias across all samples
            self.zs[i] = DMatrix::from_fn(weighted_sum.nrows(), num_samples, |row, col| {
                weighted_sum[(row, col)] + self.biases[i][(row, 0)]
            });
            
            self.activations[i + 1] = act_func.compute(&self.zs[i]);
        }

        self.activations.last().unwrap().clone()
    }

    fn init_gradients(&self) -> (Vec<DMatrix<f64>>, Vec<DMatrix<f64>>) {
        let mut nabla_w: Vec<DMatrix<f64>> = Vec::with_capacity(self.layers);
        let mut nabla_b: Vec<DMatrix<f64>> = Vec::with_capacity(self.layers);
        
        for i in 0..self.layers {
            nabla_w.push(DMatrix::zeros(self.weights[i].nrows(), self.weights[i].ncols()));
            nabla_b.push(DMatrix::zeros(self.biases[i].nrows(), self.biases[i].ncols()));
        }

        (nabla_w, nabla_b)
    }

    fn backward(
        &mut self,
        y: &DMatrix<f64>,
        learning_rate: f64,
        act_func: &ActivationFunction,
    ) {
        let batch_size = y.ncols() as f64;

        let (mut nabla_w, mut nabla_b) = self.init_gradients();
        
        let out = &self.activations[self.layers];
        let loss_grad = self.loss_function.gradient(out, y);
        let activation_deriv = act_func.derivative(&self.zs[self.layers - 1]);
        let mut delta = loss_grad.component_mul(&activation_deriv);
        
        nabla_w[self.layers - 1] = &delta * self.activations[self.layers - 1].transpose();
        nabla_b[self.layers - 1] = delta.clone();
        
        for l in (0..self.layers - 1).rev() {
            delta = self.weights[l + 1].transpose() * &delta;
            let activation_deriv = act_func.derivative(&self.zs[l]);
            delta = delta.component_mul(&activation_deriv);
            
            nabla_w[l] = &delta * self.activations[l].transpose();
            nabla_b[l] = delta.clone();
        }
        
        for l in 0..self.layers {
            self.weights[l] -= &nabla_w[l] * (learning_rate / batch_size);
            self.biases[l] -= &nabla_b[l] * (learning_rate / batch_size);
        }
    }

    pub fn train(
        &mut self,
        input: &DMatrix<f64>,
        label: &DMatrix<f64>,
        params: Hyperparams,
        act_func: &ActivationFunction,
    ) {

        for e in 0..params.epochs {
            let col_input: Vec<f64> = input.column(e % 4).iter().cloned().collect();
            let x = DMatrix::from_vec(input.nrows(), 1, col_input);

            let col_label: Vec<f64> = label.column(e % 4).iter().cloned().collect();
            let y = DMatrix::from_vec(label.nrows(), 1, col_label);

            let out = self.forward(&x, act_func);
            self.cost = self.loss_function.compute(&out, &y);
            self.backward(&y, params.learning_rate, &act_func);

            if e % 100 == 0 {
                println!("Epoch: {},  cost: {}", e, self.cost);
            }
        }        
    }

    pub fn predict(&mut self, x: &DMatrix<f64>, act_func: &ActivationFunction) -> DMatrix<f64> {
        self.forward(x, act_func)
    }
}
