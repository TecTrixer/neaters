// TODO: remove after finishing neuralnetwork
#[allow(dead_code)]
mod neuralnetwork;
#[allow(dead_code)]
mod solver;
pub use neuralnetwork::NeuralNetwork;
pub use solver::Solver;
#[cfg(test)]
mod tests;
