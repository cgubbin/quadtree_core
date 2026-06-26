use crate::{evolution::EvolutionError, scaling::ScalerError, tree::TreeError};
use trellis_runner::{EngineFailure, TrellisFloat, UserState};

#[derive(thiserror::Error, Debug)]
pub enum QuadTreeError<S, T>
where
    S: UserState,
    <S as UserState>::Float: TrellisFloat,
{
    #[error(transparent)]
    Scaling(#[from] ScalerError<T>),

    #[error(transparent)]
    Tree(#[from] TreeError<T>),

    #[error(transparent)]
    Evolution(#[from] EvolutionError<T>),

    #[error("engine failure: {0}")]
    Engine(#[from] EngineFailure<S, EvolutionError<T>>),
}
