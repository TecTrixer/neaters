# NEATeRS

**NEATeRS** is a library for training a genetic neural net through reinforcement learning.

It uses the **NEAT** algorithm developed by Ken Stanley which enables the neural net to evolve its own topology. As a result it is not necessary to know the right amount of hidden nodes as the algorithm adds them as needed using evolution with mutation. However it will also take longer to fit to a solution as it does not use prior data and it does not improve the network with backpropagation.

Nowadays there is also **HyperNEAT** which is an approach more suitable for larger problems and networks. It is still an active field of research and therefore we implemented the standard NEAT algorithm.

## Schedule

This is still only an overview of the library and the alpha version is still under development. It is **NOT** ready to use, not even for any experimental beta testers, as it does not work yet.

## Usage Example

```rust
use neaters::{Solver, NeuralNetwork};

fn main() {
	// Create a new problem solver for a problem with 2 input nodes and one output node:
	let mut solver: Solver = Solver::with_size(2, 1);

	// Train for 200 generations
	for i in 1..=200 {
		// For each neural network in the current generation run and evaluate them by assigning a fitness score:
		for nn in solver.neural_nets() {
			let input: Vec<f64> = vec![0.2, 0.8];
			let res: Vec<f64> = nn.compute(input);
			let fitness: f64 = compute_fitness(res);
			nn.assign_fitness(fitness);
		}
		println!("Average fitness in generation {} was {}", i, solver.average_fitness());
		solver.new_generation();
	}

	// Solver is trained, we can now use it for some tasks:
	let mut nn: NeuralNetwork = solver.best_neural_network();
	let input: Vec<f64> = vec![0.7, 0.3];
	let res = nn.compute(input);
	println!("The best neural network gave {} as its result for the given input", res[0]);
}

// simple fitness function which computes the difference to 0.5 and gives a higher fitness the closer the result is
fn compute_fitness(res: Vec<f64>) -> f64 {
	let expected_value: f64 = 0.5;
	let value: f64 = res[0];
	let mut diff: f64 = expected_value - value;
	if diff < 0 {diff = diff * -1.0;}
	if diff == 0 {return f64::MAX;}
	return 1 / diff;
}

```

## Features

- [x] creating neural networks
- [x] storing and loading networks
- [x] commented most functionality
- [ ] handling errors with io in NeuralNetwork
- [x] creating a solver
- [x] storing and loading solver
- [ ] handling errors with io in Solver
- [x] creating a phenotype
- [x] compute() function for neural network
- [ ] sanitizing input
- [ ] assign_fitness() function for neural network
- [ ] new_generation() function for solver
- [ ] average_fitness() function for solver
- [ ] best_neural_network() function for solver
- [ ] add advanced logging of stats to solver

## Implementation

- **Solver** is the main object for training. It contains several stats, all neural networks as well as the fitness of all networks and it performs the evolution process and creates the new generation of networks.
- **NeuralNetwork** is the neural network itself. It contains information about the edges and nodes and has functions to compute a result and to mutate itself.
