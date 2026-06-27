//! Adaptive refinement of an oscillatory function.
//!
//! This example refines the unit square according to a simple local error
//! estimate for
//!
//! ```text
//! f(x, y) = sin(8x) cos(8y).
//! ```
//!
//! For each cell, the oracle compares the value at the cell centre with the
//! values at the four corners. The largest centre-corner discrepancy is used as
//! the cell score.
//!
//! Cells with large scores are regions where the function varies strongly
//! across the cell, so they are good candidates for refinement.
//!
//! The example uses [`MaxWeightedScorePolicy`], which refines cells according to
//! `score * area`. This balances two competing effects:
//!
//! - large local error should be refined,
//! - but large cells should generally be prioritised over tiny cells with the
//!   same score.
//!
//! This is a representative pattern for adaptive approximation, interpolation,
//! and numerical quadrature.

use quadtree::{
    config::QuadTreeConfig,
    geometry::Rect,
    oracle::QuadOracle,
    policy::{CellScore, MaxWeightedScorePolicy},
    run_with_policy,
};

#[derive(Debug, Clone, Copy)]
struct ErrorEstimate(f64);

impl CellScore<f64> for ErrorEstimate {
    fn score(&self) -> f64 {
        self.0
    }
}

struct SineCosine;

impl SineCosine {
    fn f(x: f64, y: f64) -> f64 {
        (8.0 * x).sin() * (8.0 * y).cos()
    }
}

impl QuadOracle<f64> for SineCosine {
    type Data = ErrorEstimate;

    fn evaluate(&mut self, bounds: Rect<f64>) -> Self::Data {
        let c = bounds.centre();

        let centre = Self::f(c.x, c.y);

        let corners = [
            Self::f(bounds.x_min, bounds.y_min),
            Self::f(bounds.x_max, bounds.y_min),
            Self::f(bounds.x_min, bounds.y_max),
            Self::f(bounds.x_max, bounds.y_max),
        ];

        let max_corner_error = corners
            .into_iter()
            .map(|v| (v - centre).abs())
            .fold(0.0_f64, f64::max);

        ErrorEstimate(max_corner_error)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let domain = Rect::new(0.0, 1.0, 0.0, 1.0)?;

    let config = QuadTreeConfig::new(0.05)
        .with_max_iter(500)
        .with_max_depth(10)
        .with_max_leaves(50_000);

    let result = run_with_policy(domain, SineCosine, MaxWeightedScorePolicy, config)?;

    let max_score = result
        .iter()
        .map(|leaf| leaf.data().score())
        .fold(0.0_f64, f64::max);

    println!("leaves: {}", result.leaf_count());
    println!("max depth: {}", result.max_leaf_depth());
    println!("max score: {max_score}");
    println!("termination: {:?}", result.termination);

    Ok(())
}
