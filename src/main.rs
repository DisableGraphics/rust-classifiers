mod eucdist;

use std::{collections::HashMap, error::Error};

use maplit::hashmap;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use linfa_datasets::{iris, winequality};

use crate::eucdist::EuclideanDistanceClassifier;

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
	let datasets = hashmap![
		"Iris" => iris(), 
		"Wine Quality" => winequality()];
    let classifiers: HashMap<&str, Box<dyn Classifier>> = hashmap![
		"Euclidean distance" => Box::new(EuclideanDistanceClassifier::new()) as Box<dyn Classifier>
	];
	for (clasname, mut classifier) in classifiers {
		for (datasetname, dataset) in datasets.iter() {
			let records = dataset.records.clone().into_dimensionality()?;
			let records = records.view();
			let targets = dataset.targets.clone().into_dimensionality()?;
			let targets = targets.view();
			classifier
				.fit(records, targets);
			let classif = classifier.score(records, targets);
			println!("{}: score for dataset {}: {} ({} out of {})", clasname, datasetname, classif.0, classif.1, targets.len());
		}
	}
	Ok(())
}
