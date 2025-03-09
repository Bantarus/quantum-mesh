use quantum_mesh::{
    GenesisBlock, GenesisConfig, NetworkParameters, User, Node, Point, 
    HyperbolicRegion, QuantumCrypto
};
use std::error::Error;
use std::time::{SystemTime, UNIX_EPOCH};
use libp2p::identity;

async fn create_test_validator(id: &str, x: f64, y: f64) -> Result<User, Box<dyn Error>> {
    let node = Node {
        position: Point { x, y },
        connections: vec![],
        transactions: vec![],
        balance: 1000.0,
        is_validator: true,
    };

    Ok(User::new(
        id.to_string(),
        identity::Keypair::generate_ed25519(),
        node,
    )?)
}

async fn setup_test_config() -> Result<GenesisConfig, Box<dyn Error>> {
    // Create test validators at different positions
    let validators = vec![
        create_test_validator("validator1", 0.1, 0.1).await?,
        create_test_validator("validator2", -0.2, 0.3).await?,
        create_test_validator("validator3", 0.3, -0.2).await?,
    ];

    // Create test regions
    let regions = vec![
        HyperbolicRegion::new(Point { x: 0.0, y: 0.0 }, 0.5),
        HyperbolicRegion::new(Point { x: 0.5, y: 0.5 }, 0.3),
    ];

    Ok(GenesisConfig {
        initial_validators: validators,
        initial_regions: regions,
        network_parameters: NetworkParameters {
            max_validators: 100,
            min_stake: 1000.0,
            initial_supply: 1_000_000.0,
            region_count: 4,
            tessellation_density: 0.1,
        },
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    })
}

#[tokio::test]
async fn test_genesis_block_creation() -> Result<(), Box<dyn Error>> {
    // Create test configuration
    let config = setup_test_config().await?;

    // Create genesis block
    let genesis = GenesisBlock::create(config).await?;
    
    // Validate genesis block
    assert!(genesis.validate()?);
    
    // Verify initial state
    assert!(!genesis.initial_state.validator_positions.is_empty());
    assert!(!genesis.initial_state.region_boundaries.is_empty());
    assert!(!genesis.initial_state.initial_tessellation.is_empty());
    
    // Verify block properties
    assert_eq!(genesis.block.height, 0);
    assert_eq!(genesis.block.previous_hash, "0".repeat(64));
    assert!(genesis.block.transactions.is_empty());
    
    Ok(())
}

#[tokio::test]
async fn test_genesis_block_validation() -> Result<(), Box<dyn Error>> {
    let config = setup_test_config().await?;
    let genesis = GenesisBlock::create(config).await?;

    // Test hash validation
    assert!(genesis.validate()?);

    // Test validator positions
    for (_, pos) in &genesis.initial_state.validator_positions {
        assert!(pos.x >= -1.0 && pos.x <= 1.0); // Within Poincaré disk
        assert!(pos.y >= -1.0 && pos.y <= 1.0);
    }

    // Test region distribution
    assert!(genesis.initial_state.region_boundaries.len() >= 1);
    
    // Test tessellation
    assert!(!genesis.initial_state.initial_tessellation.is_empty());

    Ok(())
}

#[tokio::test]
async fn test_genesis_proof_verification() -> Result<(), Box<dyn Error>> {
    let config = setup_test_config().await?;
    let genesis = GenesisBlock::create(config).await?;
    
    // Verify the geometric proof
    let crypto = QuantumCrypto::new()?;
    let block_data = bincode::serialize(&genesis.block)?;
    
    assert!(crypto.verify_geometric_proof(
        &genesis.proof,
        &block_data
    )?);
    
    Ok(())
} 