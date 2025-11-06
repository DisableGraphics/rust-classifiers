mod eucdist;
mod mahalanobis;
mod qda;
extern crate openblas_src;

use std::{collections::BTreeMap, error::Error, fs::File, io::{stdout, Write}, path::Path};

use linfa::{Dataset, prelude::{ConfusionMatrix, ToConfusionMatrix}};
use maplit::btreemap;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis, Ix1};
use linfa_datasets::{iris, winequality, self};
use ndarray_csv::Array2Reader;
use ndarray_rand::{rand::{seq::SliceRandom, thread_rng}, rand_distr::{Distribution, Normal}};
use statrs::distribution::DiscreteUniform;
use linfa_datasets::generate::make_dataset;

use crate::{eucdist::EuclideanDistanceClassifier, mahalanobis::MahalanobisDistanceClassifier, qda::Qda};

trait Classifier {
	fn fit(&mut self, data: ArrayView2<f64>, targets: ArrayView1<usize>);
	fn decision_function(&self, data: ArrayView2<f64>) -> Array2<f64>;
	fn predict(&self, data: ArrayView2<f64>) -> Array1<usize>;
	fn score(&self, data: ArrayView2<f64>, targets: ArrayView1<usize>) -> ConfusionMatrix<usize> {
		let preds = self.predict(data);
		assert_eq!(preds.len(), targets.len(), "Vectors must have the same length");
		preds.confusion_matrix(&targets).unwrap()
	}
}

fn dataset_from_file<P: AsRef<Path>>(p: P) -> Result<Dataset<f64, usize, Ix1>, Box<dyn Error>> {
	let file = File::open(p)?;
	let mut reader = csv::ReaderBuilder::new().has_headers(true).from_reader(file);
	let array: Array2<f64> = reader.deserialize_array2_dynamic()?;

	let n_features = array.shape()[1] - 1;
    let (features, targets) = (
        array.slice(s![.., 0..n_features]).to_owned(),
        array.slice(s![.., n_features]).iter().map(|x| *x as usize).collect(),
    );
	Ok(Dataset::new(features, targets))
}

fn random_dataset_1() -> Result<Dataset<f64, usize, Ix1>, Box<dyn Error>> {
	let feat_distr = Normal::new(0.5, 5. )?;
	let target_distr = DiscreteUniform::new(0, 5)?;
	let dataset = make_dataset(512, 5, 2, feat_distr, target_distr);
	let dataset: Dataset<f64, usize, Ix1> = Dataset::new(
		dataset.records, dataset.targets.index_axis(Axis(1), 0).iter().map(|x| *x as usize).collect::<Array1<usize>>());
	Ok(dataset)
}

fn random_dataset_2() -> Result<Dataset<f64, usize, Ix1>, Box<dyn Error>> {
	let samples_per_class = 128; // 512 total
	let n_features = 5;
	let mut rng = thread_rng();

	// Define a different normal distribution per class (mean, std)
	let class_params = [(-5.0, 1.0), (0.0, 1.5), (5.0, 1.0), (10.0, 2.0)];
	let n_classes = n_features;

	let mut records = Array2::<f64>::zeros((n_classes * samples_per_class, n_features));
	let mut targets = Array1::<usize>::zeros(n_classes * samples_per_class);

	// Generate samples
	for (class_idx, (mean, std)) in class_params.iter().enumerate() {
		let normal = Normal::new(*mean, *std)?;
		for i in 0..samples_per_class {
			let sample_idx = class_idx * samples_per_class + i;
			for f in 0..n_features {
				records[[sample_idx, f]] = normal.sample(&mut rng);
			}
			targets[sample_idx] = class_idx;
		}
	}

	// ---- SHUFFLE THE DATASET ----
	let mut indices: Vec<usize> = (0..records.nrows()).collect();
	indices.shuffle(&mut rng);

	let shuffled_records = Array2::from_shape_fn(records.raw_dim(), |(i, j)| {
		let orig_i = indices[i];
		records[[orig_i, j]]
	});
	let shuffled_targets =
		Array1::from_shape_fn(targets.len(), |i| targets[indices[i]]);

	Ok(Dataset::new(shuffled_records, shuffled_targets))

}

type InnerDataset = Dataset<f64, usize, Ix1>;
type InnerDatasetTuple = (InnerDataset, InnerDataset);

fn load_datasets() -> Result<BTreeMap<&'static str, InnerDatasetTuple>, Box<dyn Error>> {
	const SPLIT_RATIO: f32 = 0.8;
	let datasets = btreemap![
		"Iris" => iris().split_with_ratio(SPLIT_RATIO), 
		"Wine Quality" => winequality().split_with_ratio(SPLIT_RATIO),
		"Random Dataset #1" => random_dataset_1()?.split_with_ratio(SPLIT_RATIO),
		"Random Dataset #2" => random_dataset_2()?.split_with_ratio(SPLIT_RATIO),
		"Test CSV dataset" => dataset_from_file("datasets/dataset.csv")?.split_with_ratio(SPLIT_RATIO),
		"Clickbait" => dataset_from_file("datasets/clickbait_title_classification.csv")?.split_with_ratio(SPLIT_RATIO),
		];
	Ok(datasets)
}

fn main() -> Result<(), Box<dyn Error>> {
	let datasets = load_datasets()?;
	let classifiers: BTreeMap<&str, Box<dyn Classifier>> = btreemap![
		"Euclidean distance" => Box::new(EuclideanDistanceClassifier::new()) as Box<dyn Classifier>,
		"Mahalanobis distance" => Box::new(MahalanobisDistanceClassifier::new()) as Box<dyn Classifier>,
		"Bayesian QDA" => Box::new(Qda::new()) as Box<dyn Classifier>
	];
	for (clasname, mut classifier) in classifiers {
		for (datasetname, dataset) in datasets.iter() {
			// Train the classifier
			let records_train = dataset.0.records.view().into_dimensionality()?;
			let targets_train = dataset.0.targets.view().into_dimensionality()?;
			classifier
				.fit(records_train, targets_train);
			// Now begin evaluation
			let records_eval = dataset.1.records.view().into_dimensionality()?;
			let targets_eval = dataset.1.targets.view().into_dimensionality()?;
			let classif = classifier.score(records_eval, targets_eval);
			println!("{} score for {}:", clasname, datasetname);
			println!("\tAccuracy: {}", classif.accuracy());
			println!("\tPrecision: {}", classif.precision());
			println!("\tRecall: {}", classif.recall());
			println!("\tF1 score: {}", classif.f1_score());

			let _ = stdout().flush();
		}
	}
	Ok(())
}
