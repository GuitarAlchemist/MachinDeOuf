//! # machin-ensemble
//!
//! Ensemble methods: bagging, random forest, boosting.
//!
//! TODO: Implement after decision tree is complete.

pub mod traits;

pub mod traits_impl {
    /// Ensemble model trait.
    pub trait EnsembleModel {
        fn n_estimators(&self) -> usize;
    }
}
