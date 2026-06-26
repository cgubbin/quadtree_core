use crate::tree::QuadTree;

use num_traits::Float;
use std::cmp::Ordering;

pub type DefaultPolicy = MaxWeightedScorePolicy;

/// Selects a leaf to refine.
pub trait RefinementPolicy<T, D> {
    fn choose_leaf(&self, tree: &QuadTree<T, D>) -> Option<usize>;
}

pub trait CellScore<T> {
    fn score(&self) -> T;
}

/// Refines the leaf with largest area.
#[derive(Debug, Clone, Copy, Default)]
pub struct LargestAreaPolicy;

impl<T, D> RefinementPolicy<T, D> for LargestAreaPolicy
where
    T: Float,
{
    fn choose_leaf(&self, tree: &QuadTree<T, D>) -> Option<usize> {
        tree.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.area().partial_cmp(&b.area()).unwrap_or(Ordering::Equal))
            .map(|(i, _)| i)
    }
}

/// Refines the leaf with largest diagonal length.
#[derive(Debug, Clone, Copy, Default)]
pub struct LargestDiameterPolicy;

impl<T, D> RefinementPolicy<T, D> for LargestDiameterPolicy
where
    T: Float,
{
    fn choose_leaf(&self, tree: &QuadTree<T, D>) -> Option<usize> {
        tree.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.diameter()
                    .partial_cmp(&b.diameter())
                    .unwrap_or(Ordering::Equal)
            })
            .map(|(i, _)| i)
    }
}

/// Refines the shallowest leaf first.
///
/// This produces breadth-first refinement.
#[derive(Debug, Clone, Copy, Default)]
pub struct ShallowestFirstPolicy;

impl<T, D> RefinementPolicy<T, D> for ShallowestFirstPolicy {
    fn choose_leaf(&self, tree: &QuadTree<T, D>) -> Option<usize> {
        tree.iter()
            .enumerate()
            .min_by_key(|(_, leaf)| leaf.depth())
            .map(|(i, _)| i)
    }
}

/// Refines the deepest leaf first.
///
/// This is mostly useful for tests or path-like refinement.
#[derive(Debug, Clone, Copy, Default)]
pub struct DeepestFirstPolicy;

impl<T, D> RefinementPolicy<T, D> for DeepestFirstPolicy {
    fn choose_leaf(&self, tree: &QuadTree<T, D>) -> Option<usize> {
        tree.iter()
            .enumerate()
            .max_by_key(|(_, leaf)| leaf.depth())
            .map(|(i, _)| i)
    }
}

/// Refines the leaf whose data has the largest score.
#[derive(Debug, Clone, Copy, Default)]
pub struct MaxScorePolicy;

impl<T, D> RefinementPolicy<T, D> for MaxScorePolicy
where
    T: Float,
    D: CellScore<T>,
{
    fn choose_leaf(&self, tree: &QuadTree<T, D>) -> Option<usize> {
        tree.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.data()
                    .score()
                    .partial_cmp(&b.data().score())
                    .unwrap_or(Ordering::Equal)
            })
            .map(|(i, _)| i)
    }
}

/// Refines by `score * area`.
///
/// Useful when `score` is an error density or local residual and larger cells
/// should be prioritised.
#[derive(Debug, Clone, Copy, Default)]
pub struct MaxWeightedScorePolicy;

impl<T, D> RefinementPolicy<T, D> for MaxWeightedScorePolicy
where
    T: Float,
    D: CellScore<T>,
{
    fn choose_leaf(&self, tree: &QuadTree<T, D>) -> Option<usize> {
        tree.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let sa = a.data().score() * a.area();
                let sb = b.data().score() * b.area();

                sa.partial_cmp(&sb).unwrap_or(Ordering::Equal)
            })
            .map(|(i, _)| i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{geometry::Rect, tree::QuadTree};

    #[derive(Debug, Clone, Copy)]
    struct Score(f64);

    impl CellScore<f64> for Score {
        fn score(&self) -> f64 {
            self.0
        }
    }

    fn root() -> Rect<f64> {
        Rect::new(0.0, 4.0, 0.0, 4.0).unwrap()
    }

    #[test]
    fn largest_area_policy_selects_root_initially() {
        let tree = QuadTree::new(root(), (), 8, 100);

        assert_eq!(LargestAreaPolicy.choose_leaf(&tree), Some(0));
    }

    #[test]
    fn largest_area_policy_prefers_unsplit_larger_leaf() {
        let mut tree = QuadTree::new(root(), (), 8, 100);

        tree.split_leaf(0, |_| ()).unwrap();
        tree.split_leaf(0, |_| ()).unwrap();

        let chosen = LargestAreaPolicy.choose_leaf(&tree).unwrap();
        let chosen_area = tree.iter().nth(chosen).unwrap().area();

        for leaf in tree.iter() {
            assert!(chosen_area >= leaf.area());
        }
    }

    #[test]
    fn largest_diameter_policy_selects_leaf_with_largest_diagonal() {
        let mut tree = QuadTree::new(root(), (), 8, 100);

        tree.split_leaf(0, |_| ()).unwrap();
        tree.split_leaf(0, |_| ()).unwrap();

        let chosen = LargestDiameterPolicy.choose_leaf(&tree).unwrap();
        let chosen_diameter = tree.iter().nth(chosen).unwrap().diameter();

        for leaf in tree.iter() {
            assert!(chosen_diameter >= leaf.diameter());
        }
    }

    #[test]
    fn shallowest_first_policy_selects_minimum_depth_leaf() {
        let mut tree = QuadTree::new(root(), (), 8, 100);

        tree.split_leaf(0, |_| ()).unwrap();
        tree.split_leaf(0, |_| ()).unwrap();

        let chosen = ShallowestFirstPolicy.choose_leaf(&tree).unwrap();
        let chosen_depth = tree.iter().nth(chosen).unwrap().depth();

        for leaf in tree.iter() {
            assert!(chosen_depth <= leaf.depth());
        }
    }

    #[test]
    fn deepest_first_policy_selects_maximum_depth_leaf() {
        let mut tree = QuadTree::new(root(), (), 8, 100);

        tree.split_leaf(0, |_| ()).unwrap();
        tree.split_leaf(0, |_| ()).unwrap();

        let chosen = DeepestFirstPolicy.choose_leaf(&tree).unwrap();
        let chosen_depth = tree.iter().nth(chosen).unwrap().depth();

        for leaf in tree.iter() {
            assert!(chosen_depth >= leaf.depth());
        }
    }

    #[test]
    fn max_score_policy_selects_highest_cell_score() {
        let mut tree = QuadTree::new(root(), Score(0.0), 8, 100);

        tree.split_leaf(0, |rect| {
            let c = rect.centre();

            if c.x > 2.0 && c.y > 2.0 {
                Score(10.0)
            } else {
                Score(1.0)
            }
        })
        .unwrap();

        let chosen = MaxScorePolicy.choose_leaf(&tree).unwrap();
        let chosen_score = tree.iter().nth(chosen).unwrap().data().score();

        assert_eq!(chosen_score, 10.0);
    }

    #[test]
    fn max_score_policy_works_after_repeated_refinement() {
        let mut tree = QuadTree::new(root(), Score(0.0), 8, 100);

        tree.split_leaf(0, |rect| Score(rect.centre().x + rect.centre().y))
            .unwrap();

        let chosen = MaxScorePolicy.choose_leaf(&tree).unwrap();
        tree.split_leaf(chosen, |rect| Score(rect.centre().x + rect.centre().y))
            .unwrap();

        let chosen = MaxScorePolicy.choose_leaf(&tree).unwrap();
        let chosen_score = tree.iter().nth(chosen).unwrap().data().score();

        for leaf in tree.iter() {
            assert!(chosen_score >= leaf.data().score());
        }
    }

    #[test]
    fn max_weighted_score_policy_uses_score_times_area() {
        let mut tree = QuadTree::new(root(), Score(0.0), 8, 100);

        tree.split_leaf(0, |rect| {
            let c = rect.centre();

            if c.x < 2.0 && c.y < 2.0 {
                Score(2.0)
            } else {
                Score(1.0)
            }
        })
        .unwrap();

        let chosen = MaxWeightedScorePolicy.choose_leaf(&tree).unwrap();
        let chosen_leaf = tree.iter().nth(chosen).unwrap();
        let chosen_weighted_score = chosen_leaf.data().score() * chosen_leaf.area();

        for leaf in tree.iter() {
            let weighted_score = leaf.data().score() * leaf.area();
            assert!(chosen_weighted_score >= weighted_score);
        }
    }

    #[test]
    fn max_weighted_score_policy_can_prefer_larger_lower_score_cell() {
        let mut tree = QuadTree::new(root(), Score(1.0), 8, 100);

        tree.split_leaf(0, |_| Score(1.0)).unwrap();

        // Split one quadrant into smaller cells with high raw scores.
        tree.split_leaf(0, |_| Score(3.0)).unwrap();

        let chosen = MaxWeightedScorePolicy.choose_leaf(&tree).unwrap();
        let chosen_leaf = tree.iter().nth(chosen).unwrap();

        // Remaining depth-1 leaves have area 4 and score 1 => weighted score 4.
        // Depth-2 leaves have area 1 and score 3 => weighted score 3.
        assert_eq!(chosen_leaf.depth(), 1);
    }

    #[test]
    fn policies_return_none_for_empty_tree_like_state_if_it_exists() {
        // Current QuadTree::new always creates one leaf, so this is mainly a
        // placeholder for if the storage representation later allows empty
        // trees.
    }
}
