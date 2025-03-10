use serde::{Serialize, Deserialize};
use crate::quantum_crypto::QuantumCrypto;
use libp2p::identity;
use std::sync::Arc;
use crate::geometry::Point;
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
    pub position: Point,
    pub connections: Vec<String>,
    pub transactions: Vec<Transaction>,
    pub balance: f64,
    pub is_validator: bool,
}

#[derive(Clone, Debug)]
pub struct User {
    pub id: String,
    pub keypair: identity::Keypair,
    pub node: Node,
    pub serialized_keypair: Option<SerializableKeypair>,
    pub quantum_crypto: Arc<QuantumCrypto>,
}

impl User {
    pub fn new(id: String, keypair: identity::Keypair, node: Node) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(User {
            id,
            keypair,
            node,
            serialized_keypair: None,
            quantum_crypto: Arc::new(QuantumCrypto::new()?),
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HyperbolicRegion {
    pub center: Point,
    pub radius: f64,
    pub nodes: Vec<String>,
    pub boundary_points: Vec<Point>,
}

impl HyperbolicRegion {
    pub fn new(center: Point, radius: f64) -> Self {
        let boundary_points = Self::calculate_boundary_points(&center, radius);
        HyperbolicRegion {
            center,
            radius,
            nodes: Vec::new(),
            boundary_points,
        }
    }

    fn calculate_boundary_points(center: &Point, radius: f64) -> Vec<Point> {
        let num_points = 16;
        let mut points = Vec::with_capacity(num_points);
        for i in 0..num_points {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / (num_points as f64);
            let x = center.x + radius * angle.cos();
            let y = center.y + radius * angle.sin();
            points.push(Point { x, y });
        }
        points
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Transaction {
    pub from: String,
    pub to: String,
    pub amount: f64,
    pub timestamp: u64,
    pub signature: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Block {
    pub previous_hash: String,
    pub timestamp: u64,
    pub transactions: Vec<Transaction>,
    pub height: u64,
    pub hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializableKeypair {
    pub bytes: Vec<u8>,
} 