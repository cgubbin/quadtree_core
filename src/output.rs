use crate::tree::QuadTree;
use trellis_runner::{RunSummary, Termination};

#[derive(Debug)]
pub struct QuadTreeResult<T, D> {
    pub tree: QuadTree<T, D>,
    pub summary: RunSummary<T>,
    pub termination: Termination,
}
