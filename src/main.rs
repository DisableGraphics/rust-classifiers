mod eucdist;

use std::error::Error;

use maplit::hashmap;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use linfa_datasets::{iris, winequality};

use crate::eucdist::EuclideanDistanceClassifier;

trait Classifier {
	fn fit(&mut self, data: ArrayView2<f64>, labels: ArrayView1<usize>);
	fn decision_function(&self, data: ArrayView2<f64>) -> Array2<f64>;
	fn predict(&self, data: ArrayView2<f64>) -> Array1<usize>;
	fn score(&self, data: ArrayView2<f64>, labels: ArrayView1<usize>) -> f64 {
		let preds = self.predict(data);
		assert_eq!(preds.len(), labels.len(), "Vectors must have the same length");
		let correct = preds.iter()
			.zip(labels.iter())
			.filter(|(pred, true_val)| **pred as usize == **true_val)
			.count();

    	correct as f64 / preds.len() as f64

	}
}

fn main() -> Result<(), Box<dyn Error>> {
	let datasets = hashmap![
		"Iris" => iris(), 
		"Wine Quality" => winequality()];
    let classifiers: Vec<Box<dyn Classifier>> = vec![Box::new(EuclideanDistanceClassifier::new())];
	for mut classifier in classifiers {
		for (datasetname, dataset) in datasets.iter() {
			let records = dataset.records.clone().into_dimensionality()?;
			let records = records.view();
			let labels = dataset.targets.clone().into_dimensionality()?;
			let labels = labels.view();
			classifier
				.fit(records, labels);
			println!("Score for dataset {}: {}", datasetname, classifier.score(records, labels))
		}
	}
	Ok(())
}
