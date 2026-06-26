mod cell;
pub mod config;
mod error;
mod evolution;
pub mod geometry;
pub mod oracle;
mod output;
pub mod policy;
mod scaling;
mod tree;

use crate::{
    config::QuadTreeConfig,
    error::QuadTreeError,
    evolution::{EvolutionError, QuadTreeRefiner},
    geometry::Rect,
    oracle::QuadOracle,
    output::QuadTreeResult,
    policy::{CellScore, MaxWeightedScorePolicy, RefinementPolicy},
    tree::QuadTree,
};

use num_traits::{Float, FromPrimitive};
use trellis_runner::{
    EngineFailure, GenerateBuilderFallible, MaxIterationPolicy, NoProgressPolicy,
    TargetValuePolicy, TrellisFloat,
};

pub fn run<T, O>(
    domain: Rect<T>,
    oracle: O,
    config: QuadTreeConfig<T>,
) -> Result<QuadTreeResult<T, O::Data>, QuadTreeError<QuadTree<T, O::Data>, T>>
where
    T: TrellisFloat + Float + FromPrimitive + Send + Sync + 'static,
    O: QuadOracle<T>,
    O::Data: CellScore<T> + Clone + Send + Sync + 'static,
{
    run_with_policy(domain, oracle, MaxWeightedScorePolicy, config)
}

pub fn run_with_policy<T, O, P>(
    domain: Rect<T>,
    mut oracle: O,
    policy: P,
    config: QuadTreeConfig<T>,
) -> Result<QuadTreeResult<T, O::Data>, QuadTreeError<QuadTree<T, O::Data>, T>>
where
    T: TrellisFloat + Float + FromPrimitive + Send + Sync + 'static,
    O: QuadOracle<T>,
    O::Data: CellScore<T> + Clone + Send + Sync + 'static,
    P: RefinementPolicy<T, O::Data> + Send + Sync + 'static,
{
    let root_data = oracle.evaluate(domain);

    let state = QuadTree::new(domain, root_data, config.max_depth, config.max_leaves);

    let procedure = QuadTreeRefiner::<T, P>::new(policy);

    let engine = <QuadTreeRefiner<T, P> as GenerateBuilderFallible>::build_for(procedure, oracle)
        .and_policy(MaxIterationPolicy::new(config.max_iter))
        .and_policy(TargetValuePolicy::new(
            T::zero(),
            config.target_score,
            config.target_window,
        ))
        .and_policy(NoProgressPolicy::new(
            config.no_progress_tolerance,
            config.no_progress_window,
        ))
        .with_initial_state(state)
        .finalise();

    let output = engine.run().map_err(QuadTreeError::Engine)?;

    Ok(QuadTreeResult {
        tree: output.result,
        summary: output.summary,
        termination: output.termination,
    })
}
