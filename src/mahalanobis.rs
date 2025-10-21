use std::collections::HashSet;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_linalg::Inverse;
use crate::Classifier;

pub struct MahalanobisDistanceClassifier {
    centroids: Option<Array2<f64>>,
    classes: Option<Vec<usize>>,
    inv_covs: Option<Vec<Array2<f64>>>,
}

impl MahalanobisDistanceClassifier {
    pub fn new() -> Self {
        Self { centroids: None, classes: None, inv_covs: None }
    }
}

impl Classifier for MahalanobisDistanceClassifier {
    fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<usize>) {
        let n_features = x.len_of(Axis(1));
        let classes: Vec<usize> = y.iter().copied().collect::<HashSet<_>>().into_iter().collect();

        let mut centroids = Array2::<f64>::zeros((classes.len(), n_features));
        let mut inv_covs = Vec::new();

        for &class_label in &classes {
            // select rows belonging to this class
            let mask: Vec<_> = y.iter().map(|&yi| yi == class_label).collect();
            let class_data: Vec<_> = x.outer_iter()
                .zip(mask.iter())
                .filter(|(_, m)| **m)
                .map(|(row, _)| row.to_owned())
                .collect();

            let n_samples = class_data.len();
            if n_samples < 2 {
                panic!("Class {} has fewer than 2 samples!", class_label);
            }

            let class_matrix = Array2::from_shape_vec(
                (n_samples, n_features),
                class_data.iter().flat_map(|r| r.to_vec()).collect(),
            ).unwrap();

            // mean
            let mean = class_matrix.mean_axis(Axis(0)).unwrap();
            centroids.row_mut(classes.iter().position(|&c| c == class_label).unwrap())
                .assign(&mean);

            // covariance (regularized)
            let centered = &class_matrix - &mean;
            let mut cov = centered.t().dot(&centered) / (n_samples as f64 - 1.0);

            // 🔒 Regularization to prevent singularity
            let reg = 1e-6;
            for i in 0..n_features {
                cov[[i, i]] += reg;
            }

            // inverse covariance
            let inv_cov = cov.inv().expect("Regularized covariance still not invertible!");
            inv_covs.push(inv_cov);
        }

        self.inv_covs = Some(inv_covs);
        self.centroids = Some(centroids);
        self.classes = Some(classes);
    }

    fn decision_function(&self, data: ArrayView2<f64>) -> Array2<f64> {
        let centroids = self.centroids.as_ref().expect("Model not fitted!");
        let inv_covs = self.inv_covs.as_ref().expect("Model not fitted!");
        let n_samples = data.len_of(Axis(0));
        let n_classes = centroids.len_of(Axis(0));

        let mut distances = Array2::<f64>::zeros((n_samples, n_classes));

        for (i, sample) in data.outer_iter().enumerate() {
            for (j, centroid) in centroids.outer_iter().enumerate() {
                let diff = &sample - &centroid;
                let d = diff.dot(&inv_covs[j].dot(&diff));
                // guard against numerical negatives (tiny floating error)
                distances[[i, j]] = d.max(0.0).sqrt();
            }
        }

        distances
    }

    fn predict(&self, x: ArrayView2<f64>) -> Array1<usize> {
        let distances = self.decision_function(x);
        let classes = self.classes.as_ref().expect("Model not fitted!");
        distances
            .axis_iter(Axis(0))
            .map(|row| {
                let idx = row
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0;
                classes[idx]
            })
            .collect()
    }
}
