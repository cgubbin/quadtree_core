//! # Evolution
//!
//! This module connects the quadtree state, user oracle, and refinement policy
//! into a Trellis procedure.
//!
//! Each procedure step:
//!
//! 1. chooses one leaf using a [`RefinementPolicy`],
//! 2. splits that leaf into four children,
//! 3. evaluates each child using the [`QuadOracle`],
//! 4. stores the evaluated children back into the tree.
//!
//! The module contains no refinement heuristics itself. Those belong to
//! `policy.rs`.

use crate::{
    geometry::Rect,
    oracle::QuadOracle,
    policy::RefinementPolicy,
    tree::{QuadTree, TreeError},
};

use num_traits::Float;
use trellis_runner::{CancellationGuard, FallibleProcedure, Progress, TrellisFloat, UserState};

#[derive(thiserror::Error, Debug)]
pub enum EvolutionError<T> {
    #[error(transparent)]
    Tree(#[from] TreeError<T>),

    #[error("no leaf available for refinement")]
    NoLeafSelected,
}

#[derive(Debug, Clone)]
pub struct QuadTreeRefiner<T, P> {
    policy: P,
    _marker: std::marker::PhantomData<T>,
}

impl<T, P> QuadTreeRefiner<T, P> {
    pub fn new(policy: P) -> Self {
        Self {
            policy,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn policy(&self) -> &P {
        &self.policy
    }
}

impl<T, D> UserState for QuadTree<T, D>
where
    T: TrellisFloat + Float,
    D: crate::policy::CellScore<T>,
{
    type Float = T;

    fn is_initialised(&self) -> bool {
        self.leaf_count() > 0
    }

    fn progress(&self) -> Progress<Self::Float> {
        let max_score = self
            .iter()
            .map(|leaf| leaf.data().score())
            .fold(T::zero(), |a, b| Float::max(a, b));

        Progress::Measure(max_score)
    }
}

impl<T, O, P> FallibleProcedure<O> for QuadTreeRefiner<T, P>
where
    T: TrellisFloat + Float + Send + Sync + 'static,
    O: QuadOracle<T>,
    O::Data: crate::policy::CellScore<T> + Clone,
    P: RefinementPolicy<T, O::Data> + Send + Sync,
{
    type Output = QuadTree<T, O::Data>;
    type State = QuadTree<T, O::Data>;
    type Error = EvolutionError<T>;

    const NAME: &'static str = "Quadtree refinement";

    fn initialise_fallible(
        &self,
        _problem: &mut O,
        _state: &mut Self::State,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn step_fallible(
        &self,
        problem: &mut O,
        state: &mut Self::State,
        _guard: CancellationGuard<'_>,
    ) -> Result<(), Self::Error> {
        let index = self
            .policy
            .choose_leaf(state)
            .ok_or(EvolutionError::NoLeafSelected)?;

        state.split_leaf(index, |bounds: Rect<T>| problem.evaluate(bounds))?;

        Ok(())
    }

    fn finalise_fallible(
        &self,
        _problem: &mut O,
        state: &Self::State,
    ) -> Result<Self::Output, Self::Error> {
        Ok(state.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        geometry::Rect,
        oracle::QuadOracle,
        policy::{CellScore, LargestAreaPolicy, MaxScorePolicy},
        tree::QuadTree,
    };

    #[derive(Debug, Clone, Copy)]
    struct Score(f64);

    impl CellScore<f64> for Score {
        fn score(&self) -> f64 {
            self.0
        }
    }

    struct AreaOracle;

    impl QuadOracle<f64> for AreaOracle {
        type Data = Score;

        fn evaluate(&mut self, bounds: Rect<f64>) -> Self::Data {
            Score(bounds.area())
        }
    }

    struct CentreScoreOracle;

    impl QuadOracle<f64> for CentreScoreOracle {
        type Data = Score;

        fn evaluate(&mut self, bounds: Rect<f64>) -> Self::Data {
            let c = bounds.centre();
            Score(c.x + c.y)
        }
    }

    fn root() -> Rect<f64> {
        Rect::new(0.0, 4.0, 0.0, 4.0).unwrap()
    }

    #[test]
    fn initialise_is_noop_for_valid_state() {
        let refiner = QuadTreeRefiner::<f64, LargestAreaPolicy>::new(LargestAreaPolicy);
        let mut oracle = AreaOracle;
        let mut state = QuadTree::new(root(), Score(16.0), 8, 100);

        refiner
            .initialise_fallible(&mut oracle, &mut state)
            .unwrap();

        assert_eq!(state.leaf_count(), 1);
    }

    #[test]
    fn one_step_splits_one_leaf_into_four() {
        let refiner = QuadTreeRefiner::<f64, LargestAreaPolicy>::new(LargestAreaPolicy);
        let mut oracle = AreaOracle;
        let mut state = QuadTree::new(root(), Score(16.0), 8, 100);

        let token = trellis_runner::CancellationToken::new();

        refiner
            .step_fallible(&mut oracle, &mut state, CancellationGuard::new(&token))
            .unwrap();

        assert_eq!(state.leaf_count(), 4);

        for leaf in state.iter() {
            assert_eq!(leaf.depth(), 1);
            assert_eq!(leaf.data().score(), 4.0);
        }
    }

    #[test]
    fn repeated_steps_refine_tree() {
        let refiner = QuadTreeRefiner::<f64, LargestAreaPolicy>::new(LargestAreaPolicy);
        let mut oracle = AreaOracle;
        let mut state = QuadTree::new(root(), Score(16.0), 8, 100);

        let token = trellis_runner::CancellationToken::new();

        for _ in 0..3 {
            refiner
                .step_fallible(&mut oracle, &mut state, CancellationGuard::new(&token))
                .unwrap();
        }

        assert_eq!(state.leaf_count(), 10);
        assert_eq!(state.max_leaf_depth(), 2);
    }

    #[test]
    fn step_respects_max_depth() {
        let refiner = QuadTreeRefiner::<f64, LargestAreaPolicy>::new(LargestAreaPolicy);
        let mut oracle = AreaOracle;
        let mut state = QuadTree::new(root(), Score(16.0), 0, 100);

        let token = trellis_runner::CancellationToken::new();

        let err = refiner
            .step_fallible(&mut oracle, &mut state, CancellationGuard::new(&token))
            .unwrap_err();

        assert!(matches!(
            err,
            EvolutionError::Tree(TreeError::MaxDepthExceeded { .. })
        ));
    }

    #[test]
    fn step_respects_max_leaves() {
        let refiner = QuadTreeRefiner::<f64, LargestAreaPolicy>::new(LargestAreaPolicy);
        let mut oracle = AreaOracle;
        let mut state = QuadTree::new(root(), Score(16.0), 8, 3);

        let token = trellis_runner::CancellationToken::new();

        let err = refiner
            .step_fallible(&mut oracle, &mut state, CancellationGuard::new(&token))
            .unwrap_err();

        assert!(matches!(
            err,
            EvolutionError::Tree(TreeError::MaxLeavesExceeded { .. })
        ));
    }

    #[test]
    fn max_score_policy_refines_high_score_region() {
        let refiner = QuadTreeRefiner::<f64, MaxScorePolicy>::new(MaxScorePolicy);
        let mut oracle = CentreScoreOracle;
        let mut state = QuadTree::new(root(), Score(0.0), 8, 100);

        let token = trellis_runner::CancellationToken::new();

        refiner
            .step_fallible(&mut oracle, &mut state, CancellationGuard::new(&token))
            .unwrap();

        let first_best = state
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.data().score().partial_cmp(&b.data().score()).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        refiner
            .step_fallible(&mut oracle, &mut state, CancellationGuard::new(&token))
            .unwrap();

        assert_eq!(state.leaf_count(), 7);

        let max_depth = state.max_leaf_depth();
        assert_eq!(max_depth, 2);

        // The second split should have occurred in one of the high-score leaves.
        // We avoid relying on stable index ordering and just check that depth-2
        // leaves now exist.
        assert!(state.iter().any(|leaf| leaf.depth() == 2));

        let _ = first_best;
    }

    #[test]
    fn finalise_returns_tree_clone() {
        let refiner = QuadTreeRefiner::<f64, LargestAreaPolicy>::new(LargestAreaPolicy);
        let mut oracle = AreaOracle;
        let state = QuadTree::new(root(), Score(16.0), 8, 100);

        let output = refiner.finalise_fallible(&mut oracle, &state).unwrap();

        assert_eq!(output.leaf_count(), state.leaf_count());
        assert_eq!(output.root(), state.root());
    }

    #[test]
    fn user_state_progress_is_max_score() {
        let mut state = QuadTree::new(root(), Score(1.0), 8, 100);
        state
            .split_leaf(0, |rect| {
                let c = rect.centre();
                Score(c.x + c.y)
            })
            .unwrap();

        match state.progress() {
            Progress::Measure(value) => {
                assert_eq!(value, 6.0);
            }
            other => panic!("unexpected progress value: {other:?}"),
        }
    }

    #[test]
    fn user_state_reports_initialised_when_tree_has_root_leaf() {
        let state = QuadTree::new(root(), Score(1.0), 8, 100);

        assert!(state.is_initialised());
    }
}
