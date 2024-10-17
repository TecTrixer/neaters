// TODO: remove after finishing neuralnetwork
mod config;
pub mod neuralnetwork;
mod phenotype;
mod solver;
mod species;
pub use neuralnetwork::NeuralNetwork;
pub use solver::Solver;
#[cfg(test)]
mod tests;
