//! # Cells
//!
//! This module defines the basic unit of quadtree refinement.
//!
//! A [`Cell`] represents one rectangular region of the domain together with
//! user-defined data associated with that region. The cell itself does not
//! interpret the data. This keeps the quadtree generic over many possible
//! algorithms:
//!
//! - adaptive function approximation,
//! - image subdivision,
//! - signed-distance refinement,
//! - density estimation,
//! - stochastic simulation,
//! - classification of regions.
//!
//! The only responsibilities of a cell are:
//!
//! - storing its rectangular bounds,
//! - storing its depth in the quadtree,
//! - exposing geometric queries delegated to [`Rect`],
//! - producing its four child cells by splitting its bounds into quadrants.
//!
//! Refinement policy and error estimation are intentionally kept outside this
//! module. Those belong to the tree and policy layers.

use crate::{
    geometry::{Point, Rect},
    scaling::ScalerError,
};

use num_traits::Float;

/// A rectangular quadtree cell with associated user-defined data.
///
/// `Cell<T, D>` is generic over:
///
/// - `T`: the floating point coordinate type,
/// - `D`: arbitrary data associated with the cell.
///
/// The cell does not assign semantics to `D`. In one algorithm it might store
/// an approximation error, in another it might store a classification label or
/// a Monte Carlo estimate.
#[derive(Debug, Clone)]
pub struct Cell<T, D> {
    bounds: Rect<T>,
    depth: usize,
    data: D,
}

impl<T, D> Cell<T, D> {
    /// Creates a new cell with explicit bounds, depth, and data.
    pub fn new(bounds: Rect<T>, depth: usize, data: D) -> Self {
        Self {
            bounds,
            depth,
            data,
        }
    }

    /// Returns the rectangular bounds of the cell.
    pub fn bounds(&self) -> Rect<T>
    where
        T: Copy,
    {
        self.bounds
    }

    /// Returns the depth of the cell in the quadtree.
    ///
    /// The root cell usually has depth `0`; each refinement increases depth by
    /// one.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Returns an immutable reference to the cell data.
    pub fn data(&self) -> &D {
        &self.data
    }

    /// Returns a mutable reference to the cell data.
    pub fn data_mut(&mut self) -> &mut D {
        &mut self.data
    }

    /// Consumes the cell and returns its components.
    pub fn into_parts(self) -> (Rect<T>, usize, D) {
        (self.bounds, self.depth, self.data)
    }
}

impl<T, D> Cell<T, D>
where
    T: Float,
{
    /// Returns the centre point of the cell.
    pub fn centre(&self) -> Point<T> {
        self.bounds.centre()
    }

    /// Returns the area of the cell.
    pub fn area(&self) -> T {
        self.bounds.area()
    }

    /// Returns the width of the cell.
    pub fn width(&self) -> T {
        self.bounds.width()
    }

    /// Returns the height of the cell.
    pub fn height(&self) -> T {
        self.bounds.height()
    }

    /// Returns the longer side length of the cell.
    pub fn longest_side(&self) -> T {
        self.bounds.longest_side()
    }

    /// Returns the diagonal length of the cell.
    pub fn diameter(&self) -> T {
        self.bounds.diameter()
    }

    /// Splits the cell into four children.
    ///
    /// The children cover the same region as the parent and each has depth
    /// `parent.depth + 1`.
    ///
    /// The supplied closure is called once for each child rectangle to generate
    /// that child's data.
    pub fn children<F>(&self, mut make_data: F) -> Result<[Cell<T, D>; 4], ScalerError<T>>
    where
        F: FnMut(Rect<T>) -> D,
    {
        let depth = self.depth + 1;
        let [sw, se, nw, ne] = self.bounds.quadrants()?;
        Ok([
            Cell::new(sw, depth, make_data(sw)),
            Cell::new(se, depth, make_data(se)),
            Cell::new(nw, depth, make_data(nw)),
            Cell::new(ne, depth, make_data(ne)),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{Point, Rect};

    use approx::assert_relative_eq;

    const TOL: f64 = 1e-12;

    #[test]
    fn cell_stores_bounds_depth_and_data() {
        let bounds = Rect::new(0.0, 2.0, 0.0, 4.0).unwrap();
        let cell = Cell::new(bounds, 3, "payload");

        assert_eq!(cell.bounds(), bounds);
        assert_eq!(cell.depth(), 3);
        assert_eq!(*cell.data(), "payload");
    }

    #[test]
    fn data_mut_allows_payload_update() {
        let bounds = Rect::new(0.0, 1.0, 0.0, 1.0).unwrap();
        let mut cell = Cell::new(bounds, 0, 1usize);

        *cell.data_mut() = 42;

        assert_eq!(*cell.data(), 42);
    }

    #[test]
    fn into_parts_returns_all_components() {
        let bounds = Rect::new(-1.0, 1.0, -2.0, 2.0).unwrap();
        let cell = Cell::new(bounds, 5, 99);

        let (b, depth, data) = cell.into_parts();

        assert_eq!(b, bounds);
        assert_eq!(depth, 5);
        assert_eq!(data, 99);
    }

    #[test]
    fn geometric_methods_delegate_to_bounds() {
        let bounds = Rect::new(0.0, 3.0, 0.0, 4.0).unwrap();
        let cell = Cell::new(bounds, 0, ());

        assert_relative_eq!(cell.width(), 3.0, epsilon = TOL);
        assert_relative_eq!(cell.height(), 4.0, epsilon = TOL);
        assert_relative_eq!(cell.area(), 12.0, epsilon = TOL);
        assert_relative_eq!(cell.longest_side(), 4.0, epsilon = TOL);
        assert_relative_eq!(cell.diameter(), 5.0, epsilon = TOL);

        assert_eq!(cell.centre(), Point { x: 1.5, y: 2.0 });
    }

    #[test]
    fn children_have_depth_incremented_by_one() {
        let bounds = Rect::new(0.0, 2.0, 0.0, 2.0).unwrap();
        let cell = Cell::new(bounds, 7, ());

        let children = cell.children(|_| ()).unwrap();

        for child in children {
            assert_eq!(child.depth(), 8);
        }
    }

    #[test]
    fn children_cover_parent_quadrants() {
        let bounds = Rect::new(0.0, 2.0, 0.0, 2.0).unwrap();
        let cell = Cell::new(bounds, 0, 0.0);

        let children = cell.children(|rect| rect.area()).unwrap();

        assert_eq!(children[0].bounds(), Rect::new(0.0, 1.0, 0.0, 1.0).unwrap());
        assert_eq!(children[1].bounds(), Rect::new(1.0, 2.0, 0.0, 1.0).unwrap());
        assert_eq!(children[2].bounds(), Rect::new(0.0, 1.0, 1.0, 2.0).unwrap());
        assert_eq!(children[3].bounds(), Rect::new(1.0, 2.0, 1.0, 2.0).unwrap());

        for child in children {
            assert_relative_eq!(*child.data(), 1.0, epsilon = TOL);
        }
    }
    #[test]
    fn children_total_area_equals_parent_area() {
        let bounds = Rect::new(-4.0, 4.0, -2.0, 6.0).unwrap();
        let cell = Cell::new(bounds, 0, ());

        let children = cell.children(|_| ()).unwrap();

        let total_area: f64 = children.iter().map(|child| child.area()).sum();

        assert_relative_eq!(total_area, cell.area(), epsilon = TOL);
    }

    #[test]
    fn children_are_contained_in_parent() {
        let bounds = Rect::new(-4.0, 4.0, -2.0, 6.0).unwrap();
        let cell = Cell::new(bounds, 0, ());

        let children = cell.children(|_| ()).unwrap();

        for child in children {
            let child_bounds = child.bounds();

            assert!(bounds.contains(Point {
                x: child_bounds.x_min,
                y: child_bounds.y_min,
            }));

            assert!(bounds.contains(Point {
                x: child_bounds.x_max,
                y: child_bounds.y_max,
            }));
        }
    }

    #[test]
    fn child_data_closure_receives_child_bounds() {
        let bounds = Rect::new(0.0, 4.0, 0.0, 4.0).unwrap();
        let cell = Cell::new(bounds, 0, Point { x: 0.0, y: 0.0 });

        let children = cell.children(|rect| rect.centre()).unwrap();

        assert_eq!(*children[0].data(), Point { x: 1.0, y: 1.0 });
        assert_eq!(*children[1].data(), Point { x: 3.0, y: 1.0 });
        assert_eq!(*children[2].data(), Point { x: 1.0, y: 3.0 });
        assert_eq!(*children[3].data(), Point { x: 3.0, y: 3.0 });
    }
}
