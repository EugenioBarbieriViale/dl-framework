use super::Net;
use super::functions::ActivationFunction;
use super::load_mnist::one_hot_decode;

use nalgebra::DMatrix;
use serde_json;
use std::fs::{File, create_dir, exists, read_to_string};
use std::io::{BufWriter, Write};
use std::path::Path;

impl Net {
    #[allow(unused)]
    pub fn predict_raw(&mut self, x: &DMatrix<f64>) -> DMatrix<f64> {
        self.forward(x)
    }

    #[allow(unused)]
    pub fn predict_prob(&mut self, x: &DMatrix<f64>) -> usize {
        if !matches!(
            self.act_functions[self.layers - 1],
            ActivationFunction::Softmax
        ) {
            println!("Softmax non detected, are you sure your output is a probability vector?");
        }
        one_hot_decode(&self.forward(x))
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

        println!("Saving model...");
        serde_json::to_writer(&mut writer, &self.params)?;
        writer.flush()?;
        println!("Done.");

        Ok(())
    }

    #[allow(unused)]
    pub fn load_from(&mut self, path: &Path) -> std::io::Result<()> {
        exists(path).expect(&format!("Model not found in path {:?}", path));

        println!("Loading model...");
        let params_str = read_to_string(path)?;
        self.params = serde_json::from_str(&params_str)?;
        println!("Done");

        Ok(())
    }
}
