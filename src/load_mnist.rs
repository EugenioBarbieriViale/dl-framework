use nalgebra::DMatrix;

use byteorder::{BigEndian, ReadBytesExt};
use flate2::read::GzDecoder;
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::Path;

struct MnistData {
    sizes: Vec<i32>,
    data: Vec<u8>,
}

impl MnistData {
    fn new(f: &File) -> Result<MnistData, std::io::Error> {
        let mut gz = GzDecoder::new(f);
        let mut contents: Vec<u8> = Vec::new();
        gz.read_to_end(&mut contents)?;

        let mut r = Cursor::new(&contents);
        let magic_number = r.read_i32::<BigEndian>()?;

        let mut sizes: Vec<i32> = Vec::new();
        let mut data: Vec<u8> = Vec::new();

        match magic_number {
            2049 => {
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            2051 => {
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            _ => panic!(),
        }
        r.read_to_end(&mut data)?;

        Ok(MnistData { sizes, data })
    }
}

pub struct MnistDataset {
    pub images: Vec<DMatrix<f64>>,
    pub classes: Vec<DMatrix<f64>>,
}

pub fn load_data(dataset_name: &str) -> Result<MnistDataset, std::io::Error> {
    let path = format!("{}-labels.idx1-ubyte.gz", dataset_name);
    let path = Path::new(&path);
    let label_data = &MnistData::new(&(File::open(path))?)?;

    let path = format!("{}-images.idx3-ubyte.gz", dataset_name);
    let path = Path::new(&path);
    let images_data = &MnistData::new(&(File::open(path))?)?;

    let mut images: Vec<DMatrix<f64>> = Vec::new();
    let image_shape = (images_data.sizes[1] * images_data.sizes[2]) as usize;

    for i in 0..images_data.sizes[0] as usize {
        let start = i * image_shape;
        let image_data = images_data.data[start..start + image_shape].to_vec();
        let image_data: Vec<f64> = image_data.into_iter().map(|x| x as f64 / 255.).collect();
        images.push(DMatrix::from_vec(image_shape, 1, image_data));
    }

    let classes: Vec<DMatrix<f64>> = label_data
        .data
        .clone()
        .into_iter()
        .map(|x| one_hot_encode(x))
        .collect::<Vec<DMatrix<f64>>>();

    Ok(MnistDataset { images, classes })
}

fn one_hot_encode(class: u8) -> DMatrix<f64> {
    let mut enc = DMatrix::zeros(10, 1);
    enc[(class as usize, 0)] = 1.0;
    enc
}
