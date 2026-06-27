//! Geometry-only quadtree refinement.
//!
//! This example demonstrates the simplest possible use of the crate: refining a
//! rectangular domain without using a problem-specific error estimate.
//!
//! Every cell receives the same constant score. The refinement policy is
//! [`LargestAreaPolicy`], so the solver always splits the largest remaining
//! leaf. This produces a balanced geometric subdivision of the domain.
//!
//! This example is useful for understanding:
//!
//! - how the quadtree evolves structurally,
//! - how refinement limits such as `max_iter`, `max_depth`, and `max_leaves`
//!   affect the output,
//! - and how to use `run_with_policy` with an explicit policy.
//!
//! No numerical approximation is being solved here. The oracle is deliberately
//! trivial so that the geometry of the refinement is isolated.

use quadtree::{
    config::QuadTreeConfig,
    geometry::Rect,
    oracle::QuadOracle,
    policy::{CellScore, LargestAreaPolicy},
    run_with_policy,
};

#[derive(Debug, Clone, Copy)]
struct Score(f64);

impl CellScore<f64> for Score {
    fn score(&self) -> f64 {
        self.0
    }
}

struct ConstantOracle;

impl QuadOracle<f64> for ConstantOracle {
    type Data = Score;

    fn evaluate(&mut self, _bounds: Rect<f64>) -> Self::Data {
        Score(1.0)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let domain = Rect::new(0.0, 1.0, 0.0, 1.0)?;

    let config = QuadTreeConfig::new(1e-12)
        .with_max_iter(6)
        .with_max_depth(8)
        .with_max_leaves(10_000);

    let result = run_with_policy(domain, ConstantOracle, LargestAreaPolicy, config)?;

    println!("leaves: {}", result.leaf_count());
    println!("max depth: {}", result.max_leaf_depth());
    println!("termination: {:?}", result.termination);

    Ok(())
}
