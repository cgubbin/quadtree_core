//! # Oracle
//!
//! Defines the user-provided evaluation interface for quadtree refinement.
//!
//! A quadtree algorithm maintains geometry and refinement state, but it does
//! not know what a cell “means”. The oracle supplies that problem-specific
//! interpretation by evaluating each cell and returning user-defined data.
//!
//! The returned data is stored on the cell and may later be used by refinement
//! policies, convergence checks, or output consumers.

use crate::geometry::Rect;

/// User-provided cell evaluator.
///
/// Implementors define how a rectangular region should be evaluated.
///
/// The output type `Data` is stored directly on the corresponding quadtree
/// cell. It may represent an approximation error, a classification result,
/// a density estimate, a signed-distance summary, or any other cell-local
/// quantity.
pub trait QuadOracle<T> {
    /// Data attached to each evaluated cell.
    type Data;

    /// Evaluate a cell region.
    ///
    /// This method is called when a new cell is created. Implementations may be
    /// deterministic or stochastic.
    fn evaluate(&mut self, bounds: Rect<T>) -> Self::Data;
}
