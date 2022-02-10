use crate::neuralnetwork::Node;
use crate::NeuralNetwork;
use rustc_hash::{FxHashMap, FxHasher};
use std::hash::BuildHasherDefault;

pub struct Phenotype {
    edges: Vec<Vec<(usize, f32)>>,
    node_index_map: FxHashMap<usize, usize>,
    node_value_array: Vec<f32>,
    pub topo_order: Vec<usize>,
}

impl Phenotype {
    pub fn from_nn_with_input(nn: NeuralNetwork, inputs: Vec<f32>) -> Self {
        let node_index_map = Phenotype::create_node_index_mapping(&nn.nodes);
        let mut edges: Vec<Vec<(usize, f32)>> = Vec::with_capacity(nn.nodes.len());
        for _ in 0..nn.nodes.len() {
            edges.push(Vec::new());
        }
        for edge in nn.edges {
            let from = *node_index_map.get(&edge.from).unwrap();
            let to = *node_index_map.get(&edge.to).unwrap();
            if edge.enabled {
                edges[from].push((to, edge.weight));
            }
        }
        let mut node_value_array: Vec<f32> = Vec::with_capacity(nn.nodes.len());
        let topo_order: Vec<usize> = Phenotype::create_topo_order(&edges, inputs.len());
        return Phenotype {
            node_index_map,
            edges,
            node_value_array,
            topo_order,
        };
    }

    fn create_topo_order(edges: &Vec<Vec<(usize, f32)>>, input_nodes: usize) -> Vec<usize> {
        let mut stack: Vec<usize> = Vec::new();
        // add every input node (one more than input bc of the constant) to the stack for dfs
        for i in 0..=input_nodes {
            stack.push(i);
        }
        let mut visited: Vec<bool> = Vec::with_capacity(edges.len());
        let mut order: Vec<usize> = Vec::with_capacity(edges.len());
        for _ in 0..edges.len() {
            visited.push(false);
            order.push(0);
        }
        let mut idx: usize = edges.len() - 1;
        while stack.len() > 0 {
            let elem = stack[stack.len() - 1];
            if visited[elem] {
                stack.pop();
                order[idx] = elem;
                if idx > 0 {
                    idx -= 1;
                }
                continue;
            } else {
                visited[elem] = true;
            }
            for edge in edges[elem].iter() {
                let to = edge.0;
                if !visited[to] {
                    stack.push(to);
                }
            }
        }
        return order;
    }

    fn create_node_index_mapping(nodes: &Vec<Node>) -> FxHashMap<usize, usize> {
        let mut map: FxHashMap<usize, usize> = FxHashMap::with_capacity_and_hasher(
            nodes.len(),
            BuildHasherDefault::<FxHasher>::default(),
        );
        let mut idx: usize = 0;
        for node in nodes {
            map.insert(idx, node.id);
            idx += 1;
        }
        return map;
    }
}
