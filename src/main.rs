mod eucdist;
mod mahalanobis;
extern crate openblas_src;

use std::{collections::BTreeMap, error::Error};

use maplit::btreemap;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use linfa_datasets::{iris, winequality};

use crate::{eucdist::EuclideanDistanceClassifier, mahalanobis::MahalanobisDistanceClassifier};

trait Classifier {
	fn fit(&mut self, data: ArrayView2<f64>, targets: ArrayView1<usize>);
	fn decision_function(&self, data: ArrayView2<f64>) -> Array2<f64>;
	fn predict(&self, data: ArrayView2<f64>) -> Array1<usize>;
	fn score(&self, data: ArrayView2<f64>, targets: ArrayView1<usize>) -> (f64, usize) {
		let preds = self.predict(data);
		assert_eq!(preds.len(), targets.len(), "Vectors must have the same length");
		let correct = preds.iter()
			.zip(targets.iter())
			.filter(|(pred, true_val)| **pred as usize == **true_val)
			.count();
    	(correct as f64 / preds.len() as f64, correct)
	}
}

fn main() -> Result<(), Box<dyn Error>> {
	const SPLIT_RATIO: f32 = 0.75;
	let datasets = btreemap![
		"Iris" => iris().split_with_ratio(SPLIT_RATIO), 
		"Wine Quality" => winequality().split_with_ratio(SPLIT_RATIO)];
    let classifiers: BTreeMap<&str, Box<dyn Classifier>> = btreemap![
		"Euclidean distance" => Box::new(EuclideanDistanceClassifier::new()) as Box<dyn Classifier>,
		"Mahalanobis distance" => Box::new(MahalanobisDistanceClassifier::new()) as Box<dyn Classifier>
	];
	for (clasname, mut classifier) in classifiers {
		for (datasetname, dataset) in datasets.iter() {
			let records = dataset.0.records.clone().into_dimensionality()?;
			let records = records.view();
			let targets = dataset.0.targets.clone().into_dimensionality()?;
			let targets = targets.view();
			let tests = dataset.1.records.clone().into_dimensionality()?;
			let tests = tests.view();
			classifier
				.fit(records, targets);
			let targets = dataset.1.targets.clone().into_dimensionality()?;
			let targets = targets.view();
			let classif = classifier.score(tests, targets);
			println!("{}: score for dataset {}: {:.4}% ({} out of {})", 
				clasname, datasetname, classif.0 * 100.0, classif.1, targets.len());
		}
	}
	Ok(())
}
