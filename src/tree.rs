//! # Quadtree
//!
//! This module implements the mutable state of an adaptive quadtree.
//!
//! The tree stores only the current leaf cells. Internal nodes are discarded
//! once split, since refinement algorithms typically operate only on the active
//! partition of the domain.
//!
//! Each refinement step replaces one leaf by its four children:
//!
//! ```text
//!         ┌─────────┐
//!         │         │
//!         │    •    │
//!         │         │
//!         └─────────┘
//!
//!                │
//!                ▼
//!
//!     ┌────┬────┐
//!     │    │    │
//!     ├────┼────┤
//!     │    │    │
//!     └────┴────┘
//! ```
//!
//! Consequently each refinement increases the number of leaves by three.
//!
//! The tree itself is deliberately policy-free. It provides only the operations
//! required to maintain a valid adaptive partition:
//!
//! - store the current leaf set,
//! - split a leaf,
//! - enforce maximum depth,
//! - enforce maximum leaf count.
//!
//! Selection of *which* leaf to split is handled by the refinement policy
//! implemented elsewhere.
use crate::{cell::Cell, geometry::Rect, scaling::ScalerError};

use num_traits::Float;
use trellis_runner::{Progress, TrellisFloat, UserState};

#[derive(thiserror::Error, Debug)]
pub enum TreeError<T> {
    #[error("leaf index {index} out of bounds for {len} leaves")]
    LeafIndexOutOfBounds { index: usize, len: usize },

    #[error("cannot split leaf at depth {depth}; max depth is {max_depth}")]
    MaxDepthExceeded { depth: usize, max_depth: usize },

    #[error("cannot split leaf; max leaves {max_leaves} would be exceeded")]
    MaxLeavesExceeded { max_leaves: usize },

    #[error("subdivision error")]
    Rect(#[from] ScalerError<T>),
}

/// An adaptive quadtree storing the active partition of a rectangular domain.
///
/// The tree owns only leaf cells; internal nodes are not retained.
///
/// This representation keeps refinement inexpensive and matches the needs of
/// adaptive algorithms, which repeatedly choose one active cell to subdivide.
///
/// # Invariants
///
/// The implementation maintains:
///
/// - every leaf lies within the root domain;
/// - leaves form a partition of the root domain;
/// - every leaf depth is at most `max_depth`;
/// - the total number of leaves is at most `max_leaves`.
#[derive(Debug, Clone)]
pub struct QuadTree<T, D> {
    root: Rect<T>,
    leaves: Vec<Cell<T, D>>,
    max_depth: usize,
    max_leaves: usize,
}

impl<T, D> QuadTree<T, D> {
    pub fn new(root: Rect<T>, root_data: D, max_depth: usize, max_leaves: usize) -> Self
    where
        T: Clone,
    {
        Self {
            root: root.clone(),
            leaves: vec![Cell::new(root, 0, root_data)],
            max_depth,
            max_leaves,
        }
    }

    pub fn root(&self) -> Rect<T>
    where
        T: Copy,
    {
        self.root
    }

    /// Returns an iterator over the current leaf cells.
    pub fn iter(&self) -> impl Iterator<Item = &Cell<T, D>> {
        self.leaves.iter()
    }

    /// Returns a mutable iterator over the current leaf cells.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Cell<T, D>> {
        self.leaves.iter_mut()
    }

    pub(crate) fn leaf(&self, index: usize) -> Option<&Cell<T, D>> {
        self.leaves.get(index)
    }

    pub(crate) fn leaf_mut(&mut self, index: usize) -> Option<&mut Cell<T, D>> {
        self.leaves.get_mut(index)
    }

    pub fn leaf_count(&self) -> usize {
        self.leaves.len()
    }

    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    pub fn max_leaves(&self) -> usize {
        self.max_leaves
    }
}

impl<T, D> QuadTree<T, D>
where
    T: Float,
{
    pub fn split_leaf<F>(&mut self, index: usize, make_data: F) -> Result<(), TreeError<T>>
    where
        F: FnMut(Rect<T>) -> D,
    {
        if index >= self.leaves.len() {
            return Err(TreeError::LeafIndexOutOfBounds {
                index,
                len: self.leaves.len(),
            });
        }

        let leaf = &self.leaves[index];

        if leaf.depth() >= self.max_depth {
            return Err(TreeError::MaxDepthExceeded {
                depth: leaf.depth(),
                max_depth: self.max_depth,
            });
        }

        // Splitting one leaf removes one and adds four: net +3.
        if self.leaves.len() + 3 > self.max_leaves {
            return Err(TreeError::MaxLeavesExceeded {
                max_leaves: self.max_leaves,
            });
        }

        let leaf = self.leaves.remove(index);
        let children = leaf.children(make_data)?;

        self.leaves.extend(children);

        Ok(())
    }

    pub fn largest_leaf_index(&self) -> Option<usize> {
        self.leaves
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.area()
                    .partial_cmp(&b.area())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
    }

    pub fn max_leaf_depth(&self) -> usize {
        self.leaves
            .iter()
            .map(|leaf| leaf.depth())
            .max()
            .unwrap_or(0)
    }

    pub fn total_leaf_area(&self) -> T {
        self.leaves
            .iter()
            .map(|leaf| leaf.area())
            .fold(T::zero(), |acc, area| acc + area)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{Point, Rect};
    use approx::assert_relative_eq;

    const TOL: f64 = 1e-12;

    fn root() -> Rect<f64> {
        Rect::new(0.0, 4.0, 0.0, 4.0).unwrap()
    }

    #[test]
    fn new_tree_contains_single_root_leaf() {
        let tree = QuadTree::new(root(), 42usize, 4, 100);

        assert_eq!(tree.leaf_count(), 1);
        assert_eq!(tree.root(), root());
        assert_eq!(tree.max_depth(), 4);
        assert_eq!(tree.max_leaves(), 100);

        let leaves: Vec<_> = tree.iter().collect();

        assert_eq!(leaves.len(), 1);
        assert_eq!(leaves[0].bounds(), root());
        assert_eq!(leaves[0].depth(), 0);
        assert_eq!(*leaves[0].data(), 42);
    }

    #[test]
    fn iter_mut_allows_leaf_data_update() {
        let mut tree = QuadTree::new(root(), 1usize, 4, 100);

        for leaf in tree.iter_mut() {
            *leaf.data_mut() += 10;
        }

        let values: Vec<_> = tree.iter().map(|leaf| *leaf.data()).collect();

        assert_eq!(values, vec![11]);
    }

    #[test]
    fn split_increases_leaf_count_by_three() {
        let mut tree = QuadTree::new(root(), 0usize, 4, 100);

        assert_eq!(tree.leaf_count(), 1);

        tree.split_leaf(0, |_| 1usize).unwrap();
        assert_eq!(tree.leaf_count(), 4);

        tree.split_leaf(0, |_| 2usize).unwrap();
        assert_eq!(tree.leaf_count(), 7);

        tree.split_leaf(0, |_| 3usize).unwrap();
        assert_eq!(tree.leaf_count(), 10);
    }

    #[test]
    fn split_replaces_parent_with_four_children() {
        let mut tree = QuadTree::new(root(), 0usize, 4, 100);

        tree.split_leaf(0, |_| 1usize).unwrap();

        let leaves: Vec<_> = tree.iter().collect();

        assert_eq!(leaves.len(), 4);

        for leaf in leaves {
            assert_eq!(leaf.depth(), 1);
            assert_eq!(*leaf.data(), 1);
        }
    }

    #[test]
    fn total_leaf_area_equals_root_after_one_split() {
        let mut tree = QuadTree::new(root(), (), 4, 100);

        tree.split_leaf(0, |_| ()).unwrap();

        assert_relative_eq!(tree.total_leaf_area(), root().area(), epsilon = TOL);
    }

    #[test]
    fn repeated_splits_preserve_total_area() {
        let mut tree = QuadTree::new(root(), (), 6, 100);

        tree.split_leaf(0, |_| ()).unwrap();
        tree.split_leaf(0, |_| ()).unwrap();
        tree.split_leaf(2, |_| ()).unwrap();
        tree.split_leaf(4, |_| ()).unwrap();

        assert_relative_eq!(tree.total_leaf_area(), root().area(), epsilon = TOL);
    }

    #[test]
    fn cannot_split_invalid_leaf_index() {
        let mut tree = QuadTree::new(root(), (), 4, 100);

        let err = tree.split_leaf(5, |_| ()).unwrap_err();

        assert!(matches!(
            err,
            TreeError::LeafIndexOutOfBounds { index: 5, len: 1 }
        ));
    }

    #[test]
    fn cannot_split_leaf_at_max_depth() {
        let mut tree = QuadTree::new(root(), (), 0, 100);

        let err = tree.split_leaf(0, |_| ()).unwrap_err();

        assert!(matches!(
            err,
            TreeError::MaxDepthExceeded {
                depth: 0,
                max_depth: 0
            }
        ));
    }

    #[test]
    fn cannot_exceed_max_leaf_count() {
        let mut tree = QuadTree::new(root(), (), 4, 3);

        let err = tree.split_leaf(0, |_| ()).unwrap_err();

        assert!(matches!(
            err,
            TreeError::MaxLeavesExceeded { max_leaves: 3 }
        ));
    }

    #[test]
    fn splitting_when_leaf_count_would_exactly_equal_max_is_allowed() {
        let mut tree = QuadTree::new(root(), (), 4, 4);

        tree.split_leaf(0, |_| ()).unwrap();

        assert_eq!(tree.leaf_count(), 4);
    }

    #[test]
    fn largest_leaf_index_returns_some_leaf() {
        let tree = QuadTree::new(root(), (), 4, 100);

        assert_eq!(tree.largest_leaf_index(), Some(0));
    }

    #[test]
    fn largest_leaf_index_prefers_unsplit_larger_leaf_after_refinement() {
        let mut tree = QuadTree::new(root(), (), 4, 100);

        tree.split_leaf(0, |_| ()).unwrap();

        // All leaves are equal immediately after splitting the root.
        let first_largest = tree.largest_leaf_index().unwrap();
        let largest_area = tree.leaf(first_largest).unwrap().area();

        for leaf in tree.iter() {
            assert_relative_eq!(leaf.area(), largest_area, epsilon = TOL);
        }

        tree.split_leaf(0, |_| ()).unwrap();

        let largest = tree.largest_leaf_index().unwrap();
        let largest_area = tree.leaf(largest).unwrap().area();

        // After splitting one quadrant, the remaining depth-1 leaves are larger
        // than the new depth-2 children.
        assert_relative_eq!(largest_area, 4.0, epsilon = TOL);
    }

    #[test]
    fn max_leaf_depth_tracks_refinement() {
        let mut tree = QuadTree::new(root(), (), 8, 100);

        assert_eq!(tree.max_leaf_depth(), 0);

        for expected_depth in 1..=3 {
            let deepest = tree
                .iter()
                .enumerate()
                .max_by_key(|(_, leaf)| leaf.depth())
                .map(|(i, _)| i)
                .unwrap();

            tree.split_leaf(deepest, |_| ()).unwrap();

            assert_eq!(tree.max_leaf_depth(), expected_depth);
        }
    }

    #[test]
    fn every_leaf_is_contained_in_root_after_repeated_splits() {
        let mut tree = QuadTree::new(root(), (), 8, 100);

        tree.split_leaf(0, |_| ()).unwrap();
        tree.split_leaf(0, |_| ()).unwrap();
        tree.split_leaf(2, |_| ()).unwrap();
        tree.split_leaf(4, |_| ()).unwrap();

        let root = tree.root();

        for leaf in tree.iter() {
            let bounds = leaf.bounds();

            assert!(root.contains(Point {
                x: bounds.x_min,
                y: bounds.y_min,
            }));

            assert!(root.contains(Point {
                x: bounds.x_max,
                y: bounds.y_max,
            }));
        }
    }

    #[test]
    fn split_child_data_is_generated_from_child_bounds() {
        let mut tree = QuadTree::new(root(), 0.0_f64, 4, 100);

        tree.split_leaf(0, |rect| rect.area()).unwrap();

        for leaf in tree.iter() {
            assert_relative_eq!(*leaf.data(), 4.0, epsilon = TOL);
        }
    }

    #[test]
    fn iter_reports_all_leaves_after_splits() {
        let mut tree = QuadTree::new(root(), (), 6, 100);

        tree.split_leaf(0, |_| ()).unwrap();
        tree.split_leaf(0, |_| ()).unwrap();

        let count = tree.iter().count();

        assert_eq!(count, tree.leaf_count());
        assert_eq!(count, 7);
    }

    #[test]
    fn iter_mut_reports_all_leaves_after_splits() {
        let mut tree = QuadTree::new(root(), 0usize, 6, 100);

        tree.split_leaf(0, |_| 0usize).unwrap();
        tree.split_leaf(0, |_| 0usize).unwrap();

        for leaf in tree.iter_mut() {
            *leaf.data_mut() += 1;
        }

        for leaf in tree.iter() {
            assert_eq!(*leaf.data(), 1);
        }
    }
}
