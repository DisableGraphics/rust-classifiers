use std::collections::HashSet;

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use crate::Classifier;

pub struct EuclideanDistanceClassifier {
	centroids: Option<Array2<f64>>,
	classes: Option<Vec<usize>>,
}

impl EuclideanDistanceClassifier {
	pub fn new() -> Self {
		Self {centroids: None, classes: None}
	}
}

impl Classifier for EuclideanDistanceClassifier {
	fn decision_function(&self, data: ndarray::ArrayView2<f64>) -> ndarray::Array2<f64> {
		let centroids = self.centroids.as_ref().expect("Model not fitted!");
		let n_samples = data.len_of(Axis(0));
		let n_classes = centroids.len_of(Axis(0));

		// Compute pairwise distances (Euclidean)
		let mut distances = Array2::<f64>::zeros((n_samples, n_classes));

		for (i, sample) in data.outer_iter().enumerate() {
			for (j, centroid) in centroids.outer_iter().enumerate() {
				let diff = &sample - &centroid;
				distances[[i, j]] = diff.mapv(|v| v.powi(2)).sum().sqrt();
			}
		}

		distances

	}

	fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<usize>) {
		let n_features = x.len_of(Axis(1));
		let classes: Vec<usize> = y.iter().copied().collect::<HashSet<_>>().into_iter().collect();

		let mut centroids = Array2::<f64>::zeros((classes.len(), n_features));

		for (class_idx, &class_label) in classes.iter().enumerate() {
			let mask: Vec<_> = y.iter().map(|&yi| yi == class_label).collect();
			let class_data: Vec<_> = x.outer_iter()
				.zip(mask.iter())
				.filter(|(_, m)| **m)
				.map(|(row, _)| row.to_owned())
				.collect();

			let n = class_data.len() as f64;
			let mean = class_data.iter()
				.fold(Array1::<f64>::zeros(n_features), |acc, row| acc + row)
				/ n;

			centroids.slice_mut(s![class_idx, ..]).assign(&mean);
		}

		self.centroids = Some(centroids);
		self.classes = Some(classes);
	}

	fn predict(&self, x: ArrayView2<f64>) -> Array1<usize> {
		let distances = self.decision_function(x);
		let classes = self.classes.as_ref().expect("Model not fitted!");
		distances
			.axis_iter(Axis(0))
			.map(|row| {
				let idx = row.iter()
					.enumerate()
					.min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
					.unwrap()
					.0;
				classes[idx]
			})
			.collect()
	}

}
