use std::collections::HashSet;

use ndarray::{s, Array1, Array2, ArrayView2, Axis, Zip};
use ndarray_linalg::{Inverse, Determinant};

use crate::Classifier;

pub struct Qda {
	mu: Option<Array2::<f64>>,
	sigma: Option<Vec<Array2<f64>>>,
	priors: Option<Array1<f64>>,
	classes: Option<Vec<usize>>,
}

impl Qda {
	pub fn new() -> Self {
		Self { mu: None, sigma: None, priors: None, classes: None }
	}
}

impl Classifier for Qda {
	fn fit(&mut self, data: ndarray::ArrayView2<f64>, targets: ndarray::ArrayView1<usize>) {
		let n_features = data.len_of(Axis(1));
		let classes: Vec<usize> = targets.iter().copied().collect::<HashSet<_>>().into_iter().collect();

		let mut means = Array2::<f64>::zeros((classes.len(), n_features));

		for (class_idx, &class_label) in classes.iter().enumerate() {
			let mask: Vec<_> = targets.iter().map(|&yi| yi == class_label).collect();
			let class_data: Vec<_> = data.outer_iter()
				.zip(mask.iter())
				.filter(|(_, m)| **m)
				.map(|(row, _)| row.to_owned())
				.collect();

			let n = class_data.len() as f64;
			let mean = class_data.iter()
				.fold(Array1::<f64>::zeros(n_features), |acc, row| acc + row)
				/ n;

			means.slice_mut(s![class_idx, ..]).assign(&mean);
		}

		let n_classes = classes.len();
		let mut covariances = Vec::with_capacity(n_classes);

		for (class_idx, &class_label) in classes.iter().enumerate() {
			// Mask rows for this class
			let mask: Vec<_> = targets.iter().map(|&yi| yi == class_label).collect();

			let class_data: Vec<Array1<f64>> = data.outer_iter()
				.zip(mask.iter())
				.filter(|(_, m)| **m)
				.map(|(row, _)| row.to_owned())
				.collect();

			let n = class_data.len() as f64;
			let mean = means.slice(s![class_idx, ..]);

			// Initialize covariance matrix
			let mut sigma = Array2::<f64>::zeros((n_features, n_features));

			for row in &class_data {
				let diff = row - &mean;
				// outer product: diff (n_features) × diff^T (n_features)
				let outer = diff.view().insert_axis(Axis(1)).dot(&diff.view().insert_axis(Axis(0)));
				sigma += &outer;
			}

			sigma /= n - 1.0;
			covariances.push(sigma);
		}
		let n_total = targets.len() as f64;
		let priors = classes
			.iter()
			.map(|&class_label| {
				let n_k = targets.iter().filter(|&&y| y == class_label).count() as f64;
				n_k / n_total
			})
			.collect::<Vec<f64>>();
		self.priors = Some(Array1::from(priors));
		self.mu = Some(means);
		self.sigma = Some(covariances);
		self.classes = Some(classes)

	}

	fn decision_function(&self, data: ArrayView2<f64>) -> Array2<f64> {
		let n_samples = data.nrows();
		let n_classes = self.mu.as_ref().unwrap().nrows();
		let n_features = data.ncols();

		let mut p = Array2::<f64>::zeros((n_samples, n_classes));
		let epsilon = 1e-6;

		let mu = self.mu.as_ref().unwrap();
		let sigma = self.sigma.as_ref().unwrap();
		let priors = self.priors.as_ref().unwrap();

		for class_idx in 0..n_classes {
			// Regularize covariance
			let sigma_reg = &sigma[class_idx] + &Array2::<f64>::eye(n_features) * epsilon;
			let sigma_inv = sigma_reg.inv().unwrap();
			let (_sign, logdet) = sigma_reg.sln_det().unwrap();
			let log_prior = priors[class_idx].ln();

			let class_mu = mu.slice(s![class_idx, ..]);

			// Vectorized over all samples
			Zip::from(p.slice_mut(s![.., class_idx]))
				.and(data.rows())
				.for_each(|p_cell, x| {
					let diff = &x - &class_mu;
					let exponent = -0.5 * diff.t().dot(&sigma_inv.dot(&diff));
					let log_likelihood = log_prior + exponent - 0.5 * logdet;
					*p_cell = log_likelihood.exp(); // Or keep log_likelihood if you want
				});
		}

		p
	}



	fn predict(&self, data: ndarray::ArrayView2<f64>) -> Array1<usize> {
		let distances = self.decision_function(data);
		let classes = self.classes.as_ref().expect("Model not fitted!");
		distances
			.axis_iter(Axis(0))
			.map(|row| {
				let idx = row.iter()
					.enumerate()
					.max_by(|a, b| a.1
						.partial_cmp(b.1)
						.unwrap_or_else(|| panic!("{} == {} = none", a.1, b.1)))
					.unwrap()
					.0;
				classes[idx]
			})
			.collect()
	}
}