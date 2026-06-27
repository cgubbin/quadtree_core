use crate::{geometry::Rect, scaling::Scaler2D, tree::QuadTree};

use num_traits::Float;
use trellis_runner::{RunSummary, Termination};

#[derive(Debug)]
pub struct QuadTreeResult<T, D> {
    pub(crate) tree: QuadTree<T, D>,
    pub(crate) scaler: Scaler2D<T>,
    pub summary: RunSummary<T>,
    pub termination: Termination,
}

#[derive(Debug, Clone, Copy)]
pub struct RawLeaf<'a, T, D> {
    bounds: Rect<T>,
    depth: usize,
    data: &'a D,
}

impl<'a, T, D> RawLeaf<'a, T, D>
where
    T: Copy + Float,
{
    pub fn centre(&self) -> crate::geometry::Point<T> {
        self.bounds.centre()
    }

    pub fn bounds(&self) -> Rect<T> {
        self.bounds
    }

    pub fn area(&self) -> T {
        self.bounds.area()
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    pub fn data(&self) -> &'a D {
        self.data
    }
}

impl<T, D> QuadTreeResult<T, D>
where
    T: Float + std::fmt::Debug,
{
    pub fn leaf_count(&self) -> usize {
        self.tree.leaf_count()
    }

    pub fn max_leaf_depth(&self) -> usize {
        self.tree.max_leaf_depth()
    }

    pub fn iter(&self) -> impl Iterator<Item = RawLeaf<'_, T, D>> {
        self.tree.iter().map(|leaf| {
            let raw_bounds = self
                .scaler
                .to_raw_rect(leaf.bounds())
                .expect("internal scaled tree should convert to raw bounds");

            RawLeaf {
                bounds: raw_bounds,
                depth: leaf.depth(),
                data: leaf.data(),
            }
        })
    }

    pub fn root(&self) -> Rect<T> {
        self.scaler
            .to_raw_rect(self.tree.root())
            .expect("internal scaled tree should convert to raw bounds")
    }

    pub fn total_leaf_area(&self) -> T {
        self.tree
            .iter()
            .map(|leaf| {
                self.scaler
                    .to_raw_rect(leaf.bounds())
                    .expect("internal scaled tree should convert to raw bounds")
            })
            .map(|leaf| leaf.area())
            .fold(T::zero(), |acc, area| acc + area)
    }

    pub(crate) fn scaled_tree(&self) -> &QuadTree<T, D> {
        &self.tree
    }
}
