use crate::neuralnetwork::NeuralNetwork;
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::{BufReader, Read, Write};

/// This is the main struct being used to train a network for a specific problem.
///
/// It contains multiple neural networks which it trains in multiple generations to get better.
/// You should be able to train a network by only interacting with this object and the contained
/// NeuralNetwork objects you can obtain by calling `solver.neural_nets()`.
// TODO: add usage example in here.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Solver {
    networks: Vec<NeuralNetwork>,
    network_size: (usize, usize),
    generation_size: usize,
    fitness: Vec<f32>,
    generation: usize,
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
        let fitness: Vec<f32> = Vec::with_capacity(generation_size);
        Solver {
            networks,
            network_size: (input_nodes, output_nodes),
            generation_size,
            fitness,
            generation: 0,
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

    fn create_from_bytes(bytes: Vec<u8>) -> Self {
        // TODO: handle serialization errors
        let decoded: Self = bincode::deserialize(&bytes).unwrap();
        decoded
    }
}
