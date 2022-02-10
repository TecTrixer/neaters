use crate::phenotype::Phenotype;
use bincode;
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::{BufReader, Read, Write};
/// Represents a node in the neural network with a specific id and a type (either Input, Hidden or
/// Output).
#[derive(Debug, PartialEq, Deserialize, Serialize)]
pub struct Node {
    /// The id of the node, needed to transform the network into its phenotype to compute the
    /// output.
    pub id: usize,
    /// The type of the node, one of Input, Hidden, Output:
    ///
    /// - Input nodes are the ones whose value is being set at the start of the computation.
    /// - Output nodes are the ones where the final computed values can be extracted.
    /// - Hidden nodes are the ones where the magic and computation happens. They are responsible
    /// for the creative computation.
    pub node_type: NodeType,
}

impl Node {
    /// Constructor for an input node with the given id. Used to create node objects.
    fn input_with_id(id: usize) -> Self {
        Node {
            id,
            node_type: NodeType::Input,
        }
    }
    /// Constructor for an output node with the given id. Used to create node objects.
    fn output_with_id(id: usize) -> Self {
        Node {
            id,
            node_type: NodeType::Output,
        }
    }
}

/// type of a node, one of Input, Hidden, Output
#[derive(Debug, Deserialize, PartialEq, Serialize)]
pub enum NodeType {
    /// Type of a node whose value is being set at the start of the computation. Their number is
    /// set at the creation of a neural network and cannot be changed.
    Input,
    /// Type of a node used in the middle of the neural net, can be connected to other hidden nodes
    /// as well as output nodes.
    Hidden,
    /// Output nodes are used to extract the final results of the computation. Their number is set
    /// at the creation of a neural network and cannot be changed.
    Output,
}

/// Struct used to represent an edge in the neural network. Is converted to an adjacency list in
/// the phenotype representation.
#[derive(Debug, Deserialize, PartialEq, Serialize)]
pub struct Edge {
    /// The id of the source node to which the edge origin is connected.
    pub from: usize,
    /// The id of the destination node, where the edge leads to.
    pub to: usize,
    /// The factor of the edge with which the source nodes value will be multiplied. The weight
    /// will change as the network is being trained to produce better outputs.
    pub weight: f32,
    /// A field to tell whether this edge is disabled in the current network or not. The edge might
    /// become disabled later during training.
    pub enabled: bool,
    /// The innovation number being used by the evolution algorithm to make an efficient merge of
    /// two networks possible.
    innovation: usize,
}

impl Edge {
    /// Constructor for creating a default edge with weight **1.0**. This edge is enabled and always
    /// has an innovation number of **0**.
    fn initial_from_to(from: usize, to: usize) -> Self {
        Edge {
            from,
            to,
            weight: 1.0,
            enabled: true,
            innovation: 0,
        }
    }
}

/// The structure being used to create, load and save a network as well as to compute outputs from
/// given inputs.
///
/// This is the main object which is being trained. After the trainging you can extract the best
/// instance from the solver. The solver is the only structure more high level than the neural
/// network.
#[derive(Debug, Deserialize, PartialEq, Serialize)]
pub struct NeuralNetwork {
    /// Storing a list of all nodes with their id's and their node types (Input, Hidden, Output).
    pub nodes: Vec<Node>,
    /// Storing a list of all edges with their destinations and other fields.
    ///
    /// This edge list will be converted to an adjacency list to be more efficient when computing the output.
    pub edges: Vec<Edge>,
    /// The id of the network is being used to identify the network within the solver, so an
    /// individual fitness value can be assigned to the exact network.
    pub id: usize,
    /// The size of the network, the first part is the number of input nodes and the second part is
    /// the number of output nodes.
    pub size: (usize, usize),
    // optionally store the phenotype if needed for multiple computations
    #[serde(skip)]
    pt: Option<Phenotype>,
}

impl NeuralNetwork {
    /// Constructor for a neural network with a given number of input nodes, output nodes and a
    /// given id.
    ///
    /// Note that this constructor will always add an additional input node whose value will always
    /// be **1.0** when computing the outputs. This is needed in case all inputs are **0.0** because the
    /// network should still be able to output **1.0**.
    ///
    /// The number of input and output nodes should not be **0** because otherwise it is not possible
    /// to create a network which can compute an output.
    // TODO: sanitize input (output_nodes = 0?)
    pub fn with_size_and_id(input_nodes: usize, output_nodes: usize, id: usize) -> Self {
        let mut nodes = Vec::with_capacity(input_nodes + 1);
        let mut edges = Vec::with_capacity((input_nodes + 1) * output_nodes);
        // creating input nodes + edges
        // it also creates a default input node with its input as a constant 1.0
        for i in 0..=input_nodes {
            // add input node
            nodes.push(Node::input_with_id(i));
            // for this input node add a default edge with weight 1.0 to every output node
            for j in (input_nodes + 1)..=(input_nodes + output_nodes) {
                edges.push(Edge::initial_from_to(i, j));
            }
        }
        // add output nodes
        for i in (input_nodes + 1)..=(input_nodes + output_nodes) {
            nodes.push(Node::output_with_id(i));
        }
        NeuralNetwork {
            nodes,
            edges,
            id,
            size: (input_nodes, output_nodes),
            pt: None,
        }
    }

    /// Constructor for a neural network with a given number of input nodes and output nodes.
    ///
    /// This constructor calls the **with_size_and_id** constructor with the constand id **0**.
    /// It should not be used because it is not suited well for training as an individual id is
    /// needed for that.
    // TODO: sanitize input (output_nodes = 0?)
    pub fn with_size(input_nodes: usize, output_nodes: usize) -> Self {
        NeuralNetwork::with_size_and_id(input_nodes, output_nodes, 0)
    }

    /// Function for computing the output of the network with a given input.
    ///
    /// Use this function to get the result from the network by giving it a `f32` for every input
    /// node you specified (with the number of input nodes when creating).
    ///
    /// # Example:
    ///
    /// ```
    /// use neaters::NeuralNetwork;
    /// // Creating a neural net with one input and one output.
    /// let mut nn = NeuralNetwork::with_size(1, 1);
    /// // Compute the result with an input for the first (and only) node of 0.5
    /// let res: Vec<f32> = nn.compute(vec![0.5]);
    /// // For the default neural net without training the output should be 0.4 (depends on the sigmoid function)
    /// assert_eq!(res, vec![0.4]);
    /// ```
    ///
    /// This function creates a phenotype to then compute the result and automatically caches it so
    /// it does not need to be created again.
    // TODO: sanitize input (length of input correct?)
    pub fn compute(&mut self, input: Vec<f32>) -> Vec<f32> {
        if let Some(pt) = &mut self.pt {
            pt.reset();
            pt.compute(input)
        } else {
            let mut pt = Phenotype::from_nn(self);
            let res = pt.compute(input);
            self.pt = Some(pt);
            res
        }
    }

    /// Returning the encoded byte representation of the neural network. This function is needed in
    /// order to store the network on a disk, but it should not be used by a client.
    // NOTE: should this be public?
    pub fn as_byte_representation(&self) -> Vec<u8> {
        // encode neural network as binary
        let encoded: Vec<u8> = match bincode::serialize(&self) {
            Ok(bytes) => bytes,
            // TODO: add clean error handling
            Err(_) => Vec::new(),
        };
        encoded
    }

    /// Saving the neural network at the specified address.
    ///
    /// # Example:
    /// ```
    /// use neaters::NeuralNetwork;
    /// # use tempfile::tempdir;
    /// # let dir = tempdir().unwrap();
    /// let nn = NeuralNetwork::with_size(4, 3);
    /// # {
    /// let path = "example-network.nn";
    /// # }
    /// # let file_location = dir.path().join("example-network.nn");
    /// # let path = file_location.as_path().to_str().unwrap();
    /// nn.save_as(path);
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

    /// Loading a neural network from a file.
    ///
    /// This is especially useful if you have already trained and saved your network because you
    /// can just load it into the program without having to create a new one and train it again.
    ///
    /// # Example:
    /// ```
    /// use neaters::NeuralNetwork;
    /// # use tempfile::tempdir;
    /// # let dir = tempdir().unwrap();
    /// # let nn2 = NeuralNetwork::with_size(4, 3);
    /// # {
    /// let path = "example-network.nn";
    /// # }
    /// # let file_location = dir.path().join("example-network.nn");
    /// # let path = file_location.as_path().to_str().unwrap();
    /// # nn2.save_as(path);
    /// let nn = NeuralNetwork::load_from(path);
    /// // Now you can use the network to compute some output.
    /// # dir.close().unwrap();
    /// ```
    // TODO: add compute usage example after compute functionality has been added.
    pub fn load_from(at: &str) -> Self {
        let bytes = NeuralNetwork::load_bytes_from(at);
        NeuralNetwork::create_from_bytes(bytes)
    }

    /// This function loads the raw bytes from a file at the speficied location. It should not be
    /// used directly by the user. Use `NeuralNetwork::load_from(path)` instead.
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
