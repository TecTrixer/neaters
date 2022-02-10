// TODO: remove after finishing neuralnetwork
pub mod neuralnetwork;
mod phenotype;
mod solver;
pub use neuralnetwork::NeuralNetwork;
pub use solver::Solver;
#[cfg(test)]
mod tests;
