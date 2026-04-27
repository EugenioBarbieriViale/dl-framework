use functions::ActivationFunction;
use functions::LossFunction;
use hyperparams::Hyperparams;

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use serde_json;
use std::fs::{File, create_dir, exists, read_to_string};
use std::io::{BufWriter, Write};
use std::path::Path;

pub mod functions;
pub mod hyperparams;
pub mod load_mnist;

#[derive(Serialize, Deserialize, Debug)]
pub struct NetParams {
    weights: Vec<DMatrix<f64>>,
    biases: Vec<DMatrix<f64>>,
}

impl NetParams {
    pub fn new(arch: &Vec<usize>, layers: usize) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for i in 0..layers {
            let cols = arch[i];
            let rows = arch[i + 1];

            let weight_matrix = DMatrix::<f64>::new_random(rows, cols) / 10000.0;
            let bias_matrix = DMatrix::<f64>::new_random(rows, 1) / 10000.0;

            weights.push(weight_matrix);
            biases.push(bias_matrix);
        }

        NetParams { weights, biases }
    }
}

#[derive(Debug)]
pub struct Net {
    layers: usize,

    act_functions: Vec<ActivationFunction>,
    loss_function: LossFunction,

    pub params: NetParams,
    zs: Vec<DMatrix<f64>>,
    activations: Vec<DMatrix<f64>>,

    pub cost: f64,
}

impl Net {
    pub fn new(
        arch: Vec<usize>,
        act_functions: Vec<ActivationFunction>,
        loss_function: LossFunction,
    ) -> Self {
        let layers = arch.len() - 1;

        if layers != act_functions.len() {
            panic!(
                "Network and activation functions sizes mismatch ({}, {})",
                layers,
                act_functions.len()
            );
        }

        let params = NetParams::new(&arch, layers);

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
            act_functions,
            loss_function,
            params,
            zs,
            activations,
            cost: 0.0,
        }
    }

    fn init_gradients(&self) -> (Vec<DMatrix<f64>>, Vec<DMatrix<f64>>) {
        let mut nabla_w: Vec<DMatrix<f64>> = Vec::with_capacity(self.layers);
        let mut nabla_b: Vec<DMatrix<f64>> = Vec::with_capacity(self.layers);

        for i in 0..self.layers {
            nabla_w.push(DMatrix::zeros(
                self.params.weights[i].nrows(),
                self.params.weights[i].ncols(),
            ));
            nabla_b.push(DMatrix::zeros(
                self.params.biases[i].nrows(),
                self.params.biases[i].ncols(),
            ));
        }

        (nabla_w, nabla_b)
    }

    fn forward(&mut self, x: &DMatrix<f64>) -> DMatrix<f64> {
        self.activations[0] = x.clone(); // slow

        for i in 0..self.layers {
            self.zs[i] = (&self.params.weights[i] * &self.activations[i]) + &self.params.biases[i];
            self.activations[i + 1] = self.act_functions[i].compute(&self.zs[i]);
        }

        self.activations.last().unwrap().clone() // slow
    }

    fn backward(
        &mut self,
        y: &DMatrix<f64>,
        nabla_w: &mut Vec<DMatrix<f64>>,
        nabla_b: &mut Vec<DMatrix<f64>>,
        learning_rate: f64,
    ) {
        // let batch_size = y.ncols() as f64; // slow

        // let out = &self.activations[self.layers];
        // let loss_grad = self.loss_function.gradient(out, y);
        // let activation_deriv =
        //     self.act_functions[self.layers - 1].derivative(&self.zs[self.layers - 1]);
        // let mut delta = loss_grad.component_mul(&activation_deriv);
        //
        // nabla_w[self.layers - 1] = &delta * self.activations[self.layers - 1].transpose();
        // nabla_b[self.layers - 1] = delta.clone();

        let out = &self.activations[self.layers];
        let loss_grad = self.loss_function.gradient(out, y);

        let mut delta = if matches!(
            self.act_functions[self.layers - 1],
            ActivationFunction::Softmax
        ) {
            // compute softmax
            let s = out;
            let dot: f64 = s.component_mul(&loss_grad).sum();
            s.component_mul(&(&loss_grad - DMatrix::from_element(s.nrows(), 1, dot)))
        } else {
            let activation_deriv =
                self.act_functions[self.layers - 1].derivative(&self.zs[self.layers - 1]);
            loss_grad.component_mul(&activation_deriv)
        };

        nabla_w[self.layers - 1] = &delta * self.activations[self.layers - 1].transpose();
        nabla_b[self.layers - 1] = delta.clone();

        for l in (0..self.layers - 1).rev() {
            delta = self.params.weights[l + 1].transpose() * &delta;
            let activation_deriv = self.act_functions[l].derivative(&self.zs[l]);
            delta = delta.component_mul(&activation_deriv);

            nabla_w[l] = &delta * self.activations[l].transpose();
            nabla_b[l] = delta.clone();
        }

        for l in 0..self.layers {
            // self.params.weights[l] -= &nabla_w[l] * (learning_rate / batch_size);
            // self.params.biases[l] -= &nabla_b[l] * (learning_rate / batch_size);
            self.params.weights[l] -= &nabla_w[l] * learning_rate;
            self.params.biases[l] -= &nabla_b[l] * learning_rate;
        }
    }

    pub fn train(
        &mut self,
        data: &Vec<DMatrix<f64>>,
        classes: &Vec<DMatrix<f64>>,
        params: &Hyperparams,
    ) {
        let len = data.len();
        assert_eq!(len, classes.len());
        let (mut nabla_w, mut nabla_b) = self.init_gradients();

        for e in 0..params.epochs {
            let mut c = 0.0;
            for (x, y) in data.into_iter().zip(classes.into_iter()) {
                let out = self.forward(&x);
                self.cost = self.loss_function.compute(&out, &y);
                self.backward(&y, &mut nabla_w, &mut nabla_b, params.learning_rate);
                c += 1.0;
                println!("{:.2}%: {}", c / (len as f32) * 100.0, self.cost);
            }
            println!("\n----------------------------------");
            println!("Epoch {} done with final cost: {}\n", e + 1, self.cost);
        }
    }

    #[allow(unused)]
    pub fn predict(&mut self, x: &DMatrix<f64>) -> DMatrix<f64> {
        self.forward(x)
    }

    #[allow(unused)]
    pub fn save_to(&self, path: &Path) -> std::io::Result<()> {
        let dir_path = path.parent().unwrap();
        match create_dir(dir_path) {
            Ok(_) => (),
            Err(_) => println!("Directory {:?} already exists, not creating.", dir_path),
        }

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        serde_json::to_writer(&mut writer, &self.params)?;
        writer.flush()?;

        Ok(())
    }

    #[allow(unused)]
    pub fn load_from(&self, path: &Path) -> std::io::Result<NetParams> {
        exists(path).expect(&format!("Model not found in path {:?}", path));
        let params_str = read_to_string(path)?;
        let params: NetParams = serde_json::from_str(&params_str)?;
        Ok(params)
    }
}
