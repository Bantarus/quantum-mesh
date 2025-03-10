// First declare the modules as public
pub mod types;
pub mod quantum_crypto;
pub mod genesis;
pub mod geometry;

// Re-export commonly used types
pub use quantum_crypto::{GeometricProof, QuantumCrypto};
pub use types::{User, Node, HyperbolicRegion, Block, Transaction, SerializableKeypair};
pub use genesis::{GenesisBlock, GenesisConfig, NetworkParameters};
pub use geometry::{Point, hyperbolic_distance};