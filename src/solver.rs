use crate::neuralnetwork::NeuralNetwork;
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::{BufReader, Read, Write};
use std::slice::IterMut;

/// This is the main struct being used to train a network for a specific problem.
///
/// It contains multiple neural networks which it trains in multiple generations to get better.
/// You should be able to train a network by only interacting with this object. You can obtain the contained
/// NeuralNetwork objects by calling `solver.neural_nets()`.
// TODO: add usage example in here.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Solver {
    networks: Vec<NeuralNetwork>,
    network_size: (usize, usize),
    generation_size: usize,
    generation: usize,
    species: Vec<Species>,
    distance_threshold: f32,
}

impl Solver {
    /// Constructor for the given problem with a specific input size, output size and a number of
    /// networks in one generation.
    ///
    /// This function should be used to create the global solver object.
    ///
    /// The input, output and generation sizes should all be bigger than 0 because otherwise the
    /// underlying networks cannot be constructed.
    /// # Example:
    /// ```
    /// use neaters::Solver;
    /// // Creates a solver for a problem with 4 inputs and 3 outputs. This solver uses 50 networks
    /// // per generation.
    /// let solver = Solver::with_size(4, 3, 50);
    /// ```
    // TODO: sanitize input (no generation size of 0, ...)
    pub fn with_size(input_nodes: usize, output_nodes: usize, generation_size: usize) -> Self {
        let mut networks: Vec<NeuralNetwork> = Vec::with_capacity(generation_size);
        for i in 0..generation_size {
            networks.push(NeuralNetwork::with_size_and_id(
                input_nodes,
                output_nodes,
                i,
            ));
        }
        let mut species = Vec::new();
        species.push(Species::new_with_network(NeuralNetwork::with_size(
            input_nodes,
            output_nodes,
        )));

        // TODO: make this variable configurable
        let distance_threshold = 1.0;
        Solver {
            networks,
            network_size: (input_nodes, output_nodes),
            generation_size,
            generation: 0,
            species,
            distance_threshold,
        }
    }

    /// Returning the encoded byte representation of the solver. This function is needed in
    /// order to store the solver on a disk, but it should not be used by a client.
    // NOTE: should this be public?
    pub fn as_byte_representation(&self) -> Vec<u8> {
        // encode neural network as binary
        let encoded: Vec<u8> = match bincode::serialize(&self) {
            Ok(bytes) => bytes,
            // TODO: add clean error handling
            Err(_) => vec![],
        };
        encoded
    }

    /// Saving the solver at the specified file location.
    ///
    /// Especially useful when you want to terminate your program and restart it without losing any
    /// training being done.
    ///
    /// # Example:
    /// ```
    /// use neaters::Solver;
    /// # use tempfile::tempdir;
    /// # let dir = tempdir().unwrap();
    /// let solver = Solver::with_size(4, 3, 3);
    /// # {
    /// let path = "example-solver.sv";
    /// # }
    /// # let file_location = dir.path().join("example-solver.sv");
    /// # let path = file_location.as_path().to_str().unwrap();
    /// // This function should be called when the program gets terminated.
    /// solver.save_as(path);
    /// # dir.close().unwrap();
    /// ```
    ///
    /// It is also possible to supply an absolute path instead of a relative path. Everything which
    /// is being understood by rust's `File::open("path...")` will be fine.
    pub fn save_as(&self, at: &str) {
        let encoded = self.as_byte_representation();
        // TODO: handle errors
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .open(at)
            .unwrap();
        file.write_all(&encoded).unwrap();
    }

    /// Loading a solver from a file.
    ///
    /// This is especially useful if you don't want to start training all over again after you
    /// needed to terminate your program. You can just store the solvers state when terminating and
    /// the load it back into the program when restarting.
    ///
    /// # Example:
    /// ```
    /// use neaters::Solver;
    /// # use tempfile::tempdir;
    /// # let dir = tempdir().unwrap();
    /// # let nn2 = Solver::with_size(4, 3, 3);
    /// # {
    /// let path = "example-solver.sv";
    /// # }
    /// # let file_location = dir.path().join("example-solver.sv");
    /// # let path = file_location.as_path().to_str().unwrap();
    /// # nn2.save_as(path);
    /// let solver = Solver::load_from(path);
    /// // Now you can use the solver to continue training its networks.
    /// # dir.close().unwrap();
    /// ```
    // TODO: add compute usage example after compute functionality has been added.
    pub fn load_from(at: &str) -> Self {
        let bytes = Solver::load_bytes_from(at);
        Solver::create_from_bytes(bytes)
    }

    /// This function loads the raw bytes from a file at the speficied location. It should not be
    /// used directly by the user. Use `Solver::load_from(path)` instead.
    // NOTE: should this be public?
    // NOTE: should we use BufReader or just a normal read from a File?
    pub fn load_bytes_from(at: &str) -> Vec<u8> {
        // TODO: handle io errors
        let file = OpenOptions::new().read(true).open(at).unwrap();
        let mut buf_reader = BufReader::new(file);
        let mut buffer: Vec<u8> = Vec::new();
        buf_reader.read_to_end(&mut buffer).unwrap();
        buffer
    }

    /// This function computes the average fitness of a generation and returns it.
    ///
    /// To have a useful result the fitness of each network must have been computed before.
    pub fn average_fitness(&mut self) -> f32 {
        let sum: f32 = self.networks.iter().fold(0.0, |a, x| a + x.fitness);
        sum / self.generation_size as f32
    }

    /// This function gives an iterator over all neural networks in one generation. It can be used
    /// to retrieve the networks for manual training.
    ///
    /// # Example
    /// ```rust
    /// use neaters::Solver;
    /// let mut solver = Solver::with_size(1,1,1);
    /// for nn in solver.neural_nets() {
    ///     // Get result based on some input values (0.1)
    ///     let result = nn.compute([0.1].to_vec())[0];
    ///     // Should compute fitness based on result
    ///     let fitness = 1.5;
    /// }
    /// ```
    pub fn neural_nets(&mut self) -> IterMut<NeuralNetwork> {
        self.networks.iter_mut()
    }

    /// Returns the best network of one generation to use.
    ///
    /// Note that this function should not be used multiple times as it is costly and could hurt
    /// performance.
    pub fn best_network(&mut self) -> NeuralNetwork {
        self.networks
            .clone()
            .into_iter()
            .fold(NeuralNetwork::with_size(0, 0), |a, x| {
                if a.fitness <= x.fitness {
                    x
                } else {
                    a
                }
            })
    }

    /// Create a new generation through speciation, mutation and ?
    ///
    /// 1. group networks by distance threshold (need distance function)
    /// 2. adjust distance threshold for next generation
    /// 3. compute adjusted fitness values
    /// 4. eliminate lower part of each group (proportional to sum of adjusted fitness of one group)
    /// 5. crossover between two networks
    /// 6. mutate them (disable Connection, change connection weight, add connection, ... Node ...)
    pub fn new_generation(&mut self) {
        self.clear_species();
        self.group_networks();
        self.remove_unused_species();
    }

    fn create_from_bytes(bytes: Vec<u8>) -> Self {
        // TODO: handle serialization errors
        let decoded: Self = bincode::deserialize(&bytes).unwrap();
        decoded
    }

    /// Group networks into their species
    fn group_networks(&mut self) {
        'outer: for network in self.networks.iter() {
            for species in self.species.iter_mut() {
                let dist = Solver::distance(&species.representative, network);
                if dist <= self.distance_threshold {
                    species.members.push(network.id);
                    continue 'outer;
                }
            }
            self.species
                .push(Species::new_with_network(network.clone()));
        }
    }

    /// Compute distance between two networks
    fn distance(representative: &NeuralNetwork, network: &NeuralNetwork) -> f32 {
        todo!()
    }

    fn clear_species(&mut self) {
        for species in self.species.iter_mut() {
            species.clear();
        }
    }

    fn remove_unused_species(&mut self) {
        let mut i = 0;
        while i < self.species.len() {
            if self.species[i].is_unused() {
                self.species.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Species {
    representative: NeuralNetwork,
    members: Vec<usize>,
}

impl Species {
    fn new_with_network(nn: NeuralNetwork) -> Self {
        Species {
            members: [nn.id].to_vec(),
            representative: nn,
        }
    }
    fn clear(&mut self) {
        self.members.clear();
    }

    fn is_unused(&self) -> bool {
        self.members.is_empty()
    }
}
