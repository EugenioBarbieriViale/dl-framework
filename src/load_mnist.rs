use nalgebra::DMatrix;

use std::fs::File;
use std::io::{Cursor, Read};
use flate2::read::GzDecoder;
use byteorder::{BigEndian, ReadBytesExt};

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

pub struct MnistImage {
    pub image: DMatrix<f64>,
    pub class: DMatrix<f64>,
}

pub fn load_data(dataset_name: &str) -> Result<Vec<MnistImage>, std::io::Error> {
    let filename = format!("{}-labels.idx1-ubyte.gz", dataset_name);
    let label_data = &MnistData::new(&(File::open(filename))?)?;
    let filename = format!("{}-images.idx3-ubyte.gz", dataset_name);

    let images_data = &MnistData::new(&(File::open(filename))?)?;
    let mut images: Vec<DMatrix<f64>> = Vec::new();
    let image_shape = (images_data.sizes[1] * images_data.sizes[2]) as usize;

    for i in 0..images_data.sizes[0] as usize {
        let start = i * image_shape;
        let image_data = images_data.data[start..start + image_shape].to_vec();
        let image_data: Vec<f64> = image_data.into_iter().map(|x| x as f64 / 255.).collect();
        images.push(DMatrix::from_vec(image_shape, 1, image_data));
    }

    let classifications: Vec<u8> = label_data.data.clone();
    let mut ret: Vec<MnistImage> = Vec::new();

    for (image, class) in images.into_iter().zip(classifications.into_iter()) {
        ret.push(MnistImage {
            image,
            class: one_hot_encode(class),
        })
    }

    Ok(ret)
}

fn one_hot_encode(class: u8) -> DMatrix<f64> {
    let mut enc = DMatrix::zeros(10, 1);
    enc[(class as usize, 0)] = 1.0;
    enc
}
