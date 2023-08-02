use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use linfa::prelude::*;
use ndarray::prelude::*;

use linfa_bayes::*;

#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

fn main() {
    let source = File::open("./train_data.csv").unwrap();
    let source = BufReader::new(source);

    let mut ys = Vec::with_capacity(1_500_000);
    let mut xs = Vec::with_capacity(1_500_000);

    for line in source.lines().skip(1) {
        let line = line.unwrap();

        let mut line = line.split(',');
        line.next();

        let feature = line.next().unwrap();

        let data = line.map(|v| v.parse::<u64>().unwrap() as f32);
        xs.extend(data);

        ys.push(feature.to_string());
    }

    let xs = Array2::from_shape_vec((xs.len() / 4096, 4096), xs).unwrap();
    
    let ys = Array1::from_vec(ys);
    let dataset = DatasetBase::new(xs, ys);

    let (train, check) = dataset.split_with_ratio(0.8);

    let model = MultinomialNbParams::new().fit(&train).unwrap();
    let pred = model.predict(&check);
    let cm = pred.confusion_matrix(&check).unwrap();

    println!("{cm:?}");
    println!("accuracy: {}, MCC: {}", cm.accuracy(), cm.mcc());
}
