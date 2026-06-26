use num_traits::{Float, FromPrimitive};

#[derive(Debug, Clone)]
pub struct QuadTreeConfig<T> {
    pub max_iter: usize,
    pub max_depth: usize,
    pub max_leaves: usize,
    pub target_score: T,
    pub target_window: usize,
    pub no_progress_tolerance: T,
    pub no_progress_window: usize,
}

impl<T> QuadTreeConfig<T>
where
    T: Float + FromPrimitive,
{
    pub fn new(target_score: T) -> Self {
        Self {
            max_iter: 1_000,
            max_depth: 32,
            max_leaves: 100_000,
            target_score,
            target_window: 3,
            no_progress_tolerance: T::from_f64(0.01).unwrap() * target_score,
            no_progress_window: 25,
        }
    }

    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    pub fn with_max_leaves(mut self, max_leaves: usize) -> Self {
        self.max_leaves = max_leaves;
        self
    }

    pub fn with_target_window(mut self, target_window: usize) -> Self {
        self.target_window = target_window;
        self
    }

    pub fn with_no_progress_tolerance(mut self, no_progress_tolerance: T) -> Self {
        self.no_progress_tolerance = no_progress_tolerance;
        self
    }

    pub fn with_no_progress_window(mut self, no_progress_window: usize) -> Self {
        self.no_progress_window = no_progress_window;
        self
    }
}
