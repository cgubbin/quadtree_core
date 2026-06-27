use approx::assert_relative_eq;
use quadtree::{
    config::QuadTreeConfig,
    geometry::Rect,
    oracle::QuadOracle,
    policy::{CellScore, LargestAreaPolicy, MaxWeightedScorePolicy},
    run, run_with_policy,
};

const TOL: f64 = 1e-12;

#[derive(Debug, Clone, Copy)]
struct Score(f64);

impl CellScore<f64> for Score {
    fn score(&self) -> f64 {
        self.0
    }
}

fn domain() -> Rect<f64> {
    Rect::new(0.0, 1.0, 0.0, 1.0).unwrap()
}

struct ConstantScore {
    score: f64,
}

impl QuadOracle<f64> for ConstantScore {
    type Data = Score;

    fn evaluate(&mut self, _bounds: Rect<f64>) -> Self::Data {
        Score(self.score)
    }
}

#[test]
fn output_tree_preserves_domain_area() {
    let config = QuadTreeConfig::new(1e-12)
        .with_max_iter(5)
        .with_max_depth(8)
        .with_target_window(1)
        .with_max_leaves(1_000);

    let result = run(domain(), ConstantScore { score: 1.0 }, config).unwrap();

    let total_area: f64 = result.iter().map(|leaf| leaf.area()).sum();

    assert_relative_eq!(total_area, domain().area(), epsilon = TOL);
}

struct GaussianBump {
    cx: f64,
    cy: f64,
    sigma: f64,
}

impl QuadOracle<f64> for GaussianBump {
    type Data = Score;

    fn evaluate(&mut self, bounds: Rect<f64>) -> Self::Data {
        let c = bounds.centre();

        let dx = c.x - self.cx;
        let dy = c.y - self.cy;
        let r2 = dx * dx + dy * dy;

        let value = (-r2 / (2.0 * self.sigma * self.sigma)).exp();

        Score(value)
    }
}

#[test]
fn weighted_score_policy_refines_near_high_score_region() {
    let config = QuadTreeConfig::new(1e-12)
        .with_max_iter(8)
        .with_max_depth(8)
        .with_target_window(1)
        .with_max_leaves(10_000);

    let result = run_with_policy(
        domain(),
        GaussianBump {
            cx: 0.75,
            cy: 0.75,
            sigma: 0.12,
        },
        MaxWeightedScorePolicy,
        config,
    )
    .unwrap();

    assert!(result.leaf_count() > 1);

    let deepest = result.iter().max_by_key(|leaf| leaf.depth()).unwrap();

    let c = deepest.centre();

    // The most refined region should be in the upper-right half,
    // near the high-score bump.
    assert!(c.x > 0.5);
    assert!(c.y > 0.5);
}

struct ErrorDecaysWithSize;

impl QuadOracle<f64> for ErrorDecaysWithSize {
    type Data = Score;

    fn evaluate(&mut self, bounds: Rect<f64>) -> Self::Data {
        Score(bounds.diameter())
    }
}

#[test]
fn score_target_drives_refinement_until_cells_are_small() {
    let config = QuadTreeConfig::new(0.2)
        .with_max_iter(200)
        .with_max_depth(10)
        .with_target_window(1)
        .with_max_leaves(100_000);

    let result = run(domain(), ErrorDecaysWithSize, config).unwrap();

    let max_score = result
        .iter()
        .map(|leaf| leaf.data().score())
        .fold(0.0_f64, f64::max);

    assert!(max_score <= 0.2);
}

struct Checkerboard {
    threshold_depth_score: f64,
}

impl QuadOracle<f64> for Checkerboard {
    type Data = Score;

    fn evaluate(&mut self, bounds: Rect<f64>) -> Self::Data {
        let c = bounds.centre();

        let ix = (c.x * 8.0).floor() as i32;
        let iy = (c.y * 8.0).floor() as i32;

        if (ix + iy) % 2 == 0 {
            Score(self.threshold_depth_score)
        } else {
            Score(0.0)
        }
    }
}

#[test]
fn discontinuous_score_field_produces_nonuniform_tree() {
    let config = QuadTreeConfig::new(1e-12)
        .with_max_iter(10)
        .with_max_depth(6)
        .with_target_window(1)
        .with_max_leaves(10_000);

    let result = run(
        domain(),
        Checkerboard {
            threshold_depth_score: 1.0,
        },
        config,
    )
    .unwrap();

    assert!(result.leaf_count() > 1);
    assert!(result.iter().any(|leaf| leaf.data().score() > 0.0));
    assert!(result.iter().any(|leaf| leaf.data().score() == 0.0));
}

#[test]
fn target_score_can_terminate_after_first_policy_check() {
    let config = QuadTreeConfig::new(1e-3)
        .with_max_iter(100)
        .with_max_depth(8)
        .with_max_leaves(1_000)
        .with_target_window(1);

    let result = run(domain(), ConstantScore { score: 1e-6 }, config).unwrap();

    // Trellis performs one step before the target policy observes convergence.
    assert_eq!(result.leaf_count(), 4);
    assert!(result.iter().all(|leaf| leaf.data().score() <= 1e-3));
}

#[test]
fn max_iteration_policy_limits_refinement() {
    let config = QuadTreeConfig::new(1e-12)
        .with_max_iter(3)
        .with_max_depth(8)
        .with_max_leaves(1_000);

    let result = run(domain(), ConstantScore { score: 1.0 }, config).unwrap();

    // Current Trellis semantics give four procedure steps for max_iter = 3.
    assert_eq!(result.leaf_count(), 13);
}

#[test]
fn max_depth_limit_is_reported_as_error() {
    let config = QuadTreeConfig::new(1e-12)
        .with_max_iter(100)
        .with_max_depth(1)
        .with_max_leaves(1_000);

    let err = run(domain(), ConstantScore { score: 1.0 }, config).unwrap_err();

    assert!(format!("{err:?}").contains("MaxDepthExceeded"));
}

#[test]
fn max_leaves_limit_is_reported_as_error() {
    let config = QuadTreeConfig::new(1e-12)
        .with_max_iter(100)
        .with_max_depth(8)
        .with_max_leaves(7);

    let err = run(domain(), ConstantScore { score: 1.0 }, config).unwrap_err();

    assert!(format!("{err:?}").contains("MaxLeavesExceeded"));
}

#[test]
fn largest_area_policy_refines_geometrically_even_if_scores_are_localised() {
    let config = QuadTreeConfig::new(1e-12)
        .with_max_iter(4)
        .with_max_depth(8)
        .with_max_leaves(10_000);

    let result = run_with_policy(
        domain(),
        GaussianBump {
            cx: 0.9,
            cy: 0.9,
            sigma: 0.05,
        },
        LargestAreaPolicy,
        config,
    )
    .unwrap();

    // Current Trellis semantics give five procedure steps for max_iter = 4:
    // 1 + 5 * 3 = 16 leaves.
    assert_eq!(result.leaf_count(), 16);

    // More importantly, LargestAreaPolicy should remain geometrically balanced.
    assert!(result.max_leaf_depth() <= 2);

    assert_relative_eq!(result.total_leaf_area(), domain().area(), epsilon = TOL);
}

#[test]
fn oracle_receives_raw_domain_coordinates() {
    struct CaptureRaw;

    impl QuadOracle<f64> for CaptureRaw {
        type Data = Score;

        fn evaluate(&mut self, bounds: Rect<f64>) -> Self::Data {
            // Root evaluation should see raw domain.
            assert!(bounds.x_min >= 10.0);
            assert!(bounds.x_max <= 20.0);
            assert!(bounds.y_min >= -5.0);
            assert!(bounds.y_max <= 5.0);

            Score(0.0)
        }
    }

    let domain = Rect::new(10.0, 20.0, -5.0, 5.0).unwrap();

    let config = QuadTreeConfig::new(1e-3)
        .with_max_iter(1)
        .with_max_depth(4)
        .with_max_leaves(100);

    let result = run(domain, CaptureRaw, config).unwrap();

    assert_eq!(result.root(), domain);
}
