use super::Net;
use super::functions::*;
use super::hyperparams::Hyperparams;

use nalgebra::DMatrix;
use rayon::prelude::*;
// use std::thread;

impl Net {
    pub fn forward(&mut self, x: &DMatrix<f64>) -> DMatrix<f64> {
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

        // println!("Training has started...");
        for e in 0..params.epochs {
            // let mut c = 0.0;
            for (x, y) in data.into_iter().zip(classes.into_iter()) {
                let out = self.forward(&x);
                self.cost = self.loss_function.compute(&out, &y);
                self.backward(&y, &mut nabla_w, &mut nabla_b, params.learning_rate);
                // c += 1.0;
                // println!("{:.2}%: {}", c / (len as f32) * 100.0, self.cost);
            }
            // println!("\n----------------------------------");
            // println!("Epoch {} done with final cost: {}\n", e + 1, self.cost);
        }
        // println!("Training ended.");
    }

    pub fn par_train(
        &mut self,
        data: &Vec<DMatrix<f64>>,
        classes: &Vec<DMatrix<f64>>,
        params: &Hyperparams,
    ) {
        let len = data.len();
        assert_eq!(len, classes.len());
        let (mut nabla_w, mut nabla_b) = self.init_gradients();

        // println!("Training has started...");
        for e in 0..params.epochs {
            for (x, y) in data
                .par_iter()
                .zip(classes.par_iter())
                .collect::<Vec<(&DMatrix<f64>, &DMatrix<f64>)>>()
            {
                let out = self.forward(&x);
                self.cost = self.loss_function.compute(&out, &y);
                self.backward(&y, &mut nabla_w, &mut nabla_b, params.learning_rate);
            }
        }
        // println!("Training ended.");
    }
}
