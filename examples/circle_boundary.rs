//! Refinement near an implicit boundary.
//!
//! This example uses a signed-distance function for a circle. Each cell is
//! classified by evaluating the signed distance at its four corners.
//!
//! If all corners are inside the circle or all corners are outside the circle,
//! the cell is treated as resolved and receives score zero.
//!
//! If some corners are inside and others are outside, the circle boundary is
//! assumed to cross the cell. Such cells receive a score proportional to their
//! diameter, so boundary cells continue to refine until they become small.
//!
//! This demonstrates a common quadtree use case:
//!
//! - implicit geometry,
//! - interface tracking,
//! - level-set refinement,
//! - mesh generation around boundaries.
//!
//! The result is a non-uniform tree concentrated around the circle boundary.
use quadtree_core::{
    config::QuadTreeConfig,
    geometry::{Point, Rect},
    oracle::QuadOracle,
    policy::{CellScore, MaxWeightedScorePolicy},
    run_with_policy,
};

#[derive(Debug, Clone, Copy)]
struct BoundaryScore(f64);

impl CellScore<f64> for BoundaryScore {
    fn score(&self) -> f64 {
        self.0
    }
}

struct Circle {
    centre: Point<f64>,
    radius: f64,
}

impl Circle {
    fn signed_distance(&self, p: Point<f64>) -> f64 {
        let dx = p.x - self.centre.x;
        let dy = p.y - self.centre.y;

        (dx * dx + dy * dy).sqrt() - self.radius
    }
}

impl QuadOracle<f64> for Circle {
    type Data = BoundaryScore;

    fn evaluate(&mut self, bounds: Rect<f64>) -> Self::Data {
        let corners = [
            Point {
                x: bounds.x_min,
                y: bounds.y_min,
            },
            Point {
                x: bounds.x_max,
                y: bounds.y_min,
            },
            Point {
                x: bounds.x_min,
                y: bounds.y_max,
            },
            Point {
                x: bounds.x_max,
                y: bounds.y_max,
            },
        ];

        let mut has_inside = false;
        let mut has_outside = false;

        for p in corners {
            let d = self.signed_distance(p);

            if d <= 0.0 {
                has_inside = true;
            } else {
                has_outside = true;
            }
        }

        // Refine cells crossed by the boundary. Weight by diameter so the score
        // naturally decreases as boundary cells get smaller.
        let score = if has_inside && has_outside {
            bounds.diameter()
        } else {
            0.0
        };

        BoundaryScore(score)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let domain = Rect::new(0.0, 1.0, 0.0, 1.0)?;

    let config = QuadTreeConfig::new(0.01)
        .with_max_iter(1_000)
        .with_max_depth(12)
        .with_max_leaves(100_000);

    let result = run_with_policy(
        domain,
        Circle {
            centre: Point { x: 0.5, y: 0.5 },
            radius: 0.3,
        },
        MaxWeightedScorePolicy,
        config,
    )?;

    let boundary_cells = result
        .iter()
        .filter(|leaf| leaf.data().score() > 0.0)
        .count();

    println!("leaves: {}", result.leaf_count());
    println!("boundary cells: {boundary_cells}");
    println!("max depth: {}", result.max_leaf_depth());
    println!("termination: {:?}", result.termination);

    Ok(())
}
