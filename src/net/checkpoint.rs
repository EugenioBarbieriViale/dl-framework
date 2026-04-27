use super::functions::ActivationFunction;
use super::{Net, NetParams};

use nalgebra::DMatrix;
use serde_json;
use std::fs::{File, create_dir, exists, read_to_string};
use std::io::{BufWriter, Write};
use std::path::Path;

impl Net {
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
