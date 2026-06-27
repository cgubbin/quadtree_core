//! # Adaptive Quadtree Refinement
//!
//! `quadtree` is a generic adaptive spatial subdivision library for two-dimensional
//! rectangular domains.
//!
//! The crate separates **geometry**, **evaluation**, and **refinement strategy**
//! into independent components, allowing the same refinement engine to be reused
//! across a wide range of numerical algorithms.
//!
//! Typical applications include:
//!
//! - adaptive quadrature,
//! - finite element mesh generation,
//! - adaptive interpolation,
//! - image processing,
//! - implicit geometry,
//! - level-set methods,
//! - contour and root finding,
//! - spatial error estimation,
//! - any algorithm requiring adaptive subdivision of a rectangular domain.
//!
//! The library is built on top of the
//! [`trellis_runner`](https://crates.io/crates/trellis_runner) engine, providing
//! deterministic execution, configurable stopping criteria, progress reporting,
//! and cancellation support.
//!
//! # Design
//!
//! A quadtree consists of a collection of rectangular cells covering the domain.
//! Beginning with a single root cell, the solver repeatedly
//!
//! 1. selects one leaf according to a refinement policy,
//! 2. subdivides that cell into four children,
//! 3. evaluates each child using a user-provided oracle,
//! 4. stores the resulting data on the new cells.
//!
//! This process continues until one of the configured stopping criteria is met.
//!
//! Unlike many quadtree implementations, the library makes no assumptions about
//! what the stored cell data represents. Cells may contain error estimates,
//! physical measurements, classifications, interpolation coefficients,
//! probabilities, or arbitrary user-defined state.
//!
//! # Core concepts
//!
//! Three traits define the behaviour of the solver.
//!
//! ## Oracle
//!
//! A [`QuadOracle`] evaluates newly-created cells.
//!
//! ```text
//! Rect ─────────► Oracle ─────────► Cell Data
//! ```
//!
//! The oracle encapsulates the application-specific numerical work while the
//! quadtree manages only geometry and refinement.
//!
//! ## Cell score
//!
//! The refinement engine measures progress using a scalar score supplied by the
//! cell data.
//!
//! ```text
//! Cell Data ─────────► CellScore
//! ```
//!
//! By default the global progress measure is the maximum score over all leaves.
//! Refinement therefore terminates once every leaf satisfies the requested
//! tolerance.
//!
//! ## Refinement policy
//!
//! A [`RefinementPolicy`] determines *which* leaf should be subdivided next.
//!
//! The crate provides several built-in policies, including
//!
//! - largest-area refinement,
//! - maximum score refinement,
//! - weighted score refinement.
//!
//! Custom policies can easily be implemented for application-specific
//! refinement strategies.
//!
//! # Running a refinement
//!
//! The simplest interface is [`run`], which uses the default weighted-score
//! refinement policy.
//!
//! ```no_run
//! use quadtree::{
//!     run,
//!     QuadTreeConfig,
//!     Rect,
//!     QuadOracle,
//!     CellScore,
//! };
//!
//! #[derive(Clone, Copy, Debug)]
//! struct Score(f64);
//!
//! impl CellScore<f64> for Score {
//!     fn score(&self) -> f64 {
//!         self.0
//!     }
//! }
//!
//! struct Oracle;
//!
//! impl QuadOracle<f64> for Oracle {
//!     type Data = Score;
//!
//!     fn evaluate(&mut self, bounds: Rect<f64>) -> Self::Data {
//!         Score(bounds.area())
//!     }
//! }
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let domain = Rect::new(0.0, 1.0, 0.0, 1.0)?;
//!
//! let config = QuadTreeConfig::new(0.01);
//!
//! let result = run(domain, Oracle, config)?;
//!
//! println!("leaf count = {}", result.leaf_count());
//! # Ok(())
//! # }
//! ```
//!
//! Alternative refinement strategies can be selected using
//! [`run_with_policy`].
//!
//! # Stopping criteria
//!
//! The solver uses Trellis policies to terminate refinement.
//!
//! By default these include
//!
//! - maximum iteration count,
//! - target score,
//! - lack of progress.
//!
//! Additional policies may be composed without changing the refinement
//! algorithm.
//!
//! # Examples
//!
//! The `examples/` directory contains several complete applications:
//!
//! - **geometry_only** — balanced geometric subdivision using largest-area
//!   refinement.
//! - **sine_cosine_error** — adaptive refinement driven by a local interpolation
//!   error estimate.
//! - **circle_boundary** — refinement concentrated around an implicit interface
//!   represented by a signed-distance function.
//!
//! These examples demonstrate how the same refinement engine can be reused with
//! entirely different refinement objectives.
//!
//! # Philosophy
//!
//! This crate intentionally separates concerns.
//!
//! - Geometry is handled by the quadtree.
//! - Numerical evaluation is handled by the oracle.
//! - Refinement strategy is handled by policies.
//! - Execution is handled by Trellis.
//!
//! By keeping these components independent, sophisticated adaptive algorithms
//! can be constructed by composing small, reusable building blocks rather than
//! modifying the underlying tree implementation.

#![allow(dead_code)]

mod cell;
pub mod config;
mod error;
mod evolution;
pub mod geometry;
pub mod oracle;
mod output;
pub mod policy;
mod scaling;
mod tree;

pub use {
    config::QuadTreeConfig,
    error::QuadTreeError,
    geometry::Rect,
    oracle::QuadOracle,
    output::QuadTreeResult,
    policy::{CellScore, MaxWeightedScorePolicy, RefinementPolicy},
};

use crate::{evolution::QuadTreeRefiner, scaling::Scaler2D, tree::QuadTree};

use num_traits::{Float, FromPrimitive};
use trellis_runner::{
    GenerateBuilderFallible, MaxIterationPolicy, NoProgressPolicy, TargetValuePolicy, TrellisFloat,
};

#[allow(clippy::type_complexity)]
pub fn run<T, O>(
    domain: Rect<T>,
    oracle: O,
    config: QuadTreeConfig<T>,
) -> Result<QuadTreeResult<T, O::Data>, QuadTreeError<QuadTree<T, O::Data>, T>>
where
    T: TrellisFloat + Float + FromPrimitive + Send + Sync + 'static,
    O: QuadOracle<T>,
    O::Data: CellScore<T> + Clone + Send + Sync + 'static,
{
    run_with_policy(domain, oracle, MaxWeightedScorePolicy, config)
}

#[allow(clippy::type_complexity)]
pub fn run_with_policy<T, O, P>(
    domain: Rect<T>,
    mut oracle: O,
    policy: P,
    config: QuadTreeConfig<T>,
) -> Result<QuadTreeResult<T, O::Data>, QuadTreeError<QuadTree<T, O::Data>, T>>
where
    T: TrellisFloat + Float + FromPrimitive + Send + Sync + 'static,
    O: QuadOracle<T>,
    O::Data: CellScore<T> + Clone + Send + Sync + 'static,
    P: RefinementPolicy<T, O::Data> + Send + Sync + 'static,
{
    let scaler = Scaler2D::unit_square(domain)?;

    let scaled_domain = scaler.scaled_domain();

    let root_data = oracle.evaluate(domain);

    let state = QuadTree::new(
        scaled_domain,
        root_data,
        config.max_depth,
        config.max_leaves,
    );

    let procedure = QuadTreeRefiner::<T, P>::new(policy, scaler.clone());

    let engine = <QuadTreeRefiner<T, P> as GenerateBuilderFallible>::build_for(procedure, oracle)
        .and_policy(MaxIterationPolicy::new(config.max_iter))
        .and_policy(TargetValuePolicy::new(
            T::zero(),
            config.target_score,
            config.target_window,
        ))
        .and_policy(NoProgressPolicy::new(
            config.no_progress_tolerance,
            config.no_progress_window,
        ))
        .with_initial_state(state)
        .finalise();

    let output = engine.run().map_err(QuadTreeError::Engine)?;

    Ok(QuadTreeResult {
        tree: output.result,
        scaler,
        summary: output.summary,
        termination: output.termination,
    })
}
