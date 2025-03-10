use ml_kem::{MlKem768, SharedKey, KemCore, MlKem768Params, array::Array};
use ml_kem::kem::{DecapsulationKey, EncapsulationKey, Decapsulate, Encapsulate, Kem};
use chacha20poly1305::{
    aead::{Aead, AeadCore, KeyInit, OsRng},
    ChaCha20Poly1305, Nonce, Key
};
use pqcrypto_dilithium::dilithium3::{
    keypair, 
    PublicKey, 
    SecretKey,
    detached_sign,
    verify_detached_signature
};
use pqcrypto_traits::sign::DetachedSignature;
use rand::thread_rng;
use std::sync::Arc;
use std::fmt;
use std::error::Error as StdError;
use serde::{Serialize, Deserialize};
use sha2::{Sha256, Digest};
use std::time::{SystemTime, UNIX_EPOCH};
use std::collections::HashMap;
use std::sync::Mutex;
use crate::geometry::{Point, hyperbolic_distance};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeometricProof {
    pub tessellation_points: Vec<Point>,
    pub verification_path: Vec<Point>,
    pub position_data: Vec<u8>,
    pub hash: String,
    pub signature: Vec<u8>,
    pub timestamp: u64,
}

/// Represents the quantum-resistant cryptographic state for a node
#[derive(Clone)]
pub struct QuantumCrypto {
    // Decapsulation key is private and used for receiving encrypted messages
    decapsulation_key: Arc<DecapsulationKey<MlKem768Params>>,
    // Encapsulation key is public and shared with other nodes
    encapsulation_key: Arc<EncapsulationKey<MlKem768Params>>,
    
    // New fields for signatures
    signing_secret_key: SecretKey,
    signing_public_key: PublicKey,
    
    // Cache for shared keys
    shared_keys: Arc<Mutex<HashMap<String, (SharedKey<Kem<MlKem768Params>>, SystemTime)>>>,
}

impl std::fmt::Debug for QuantumCrypto {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantumCrypto")
            .field("decapsulation_key", &"<redacted>")
            .field("encapsulation_key", &"<redacted>")
            .field("signing_secret_key", &"<redacted>")
            .field("signing_public_key", &"<redacted>")
            .field("shared_keys", &self.shared_keys)
            .finish()
    }
}

#[derive(Debug)]
pub enum QuantumCryptoError {
    EncapsulationError,
    DecapsulationError,
    KeyGenerationError,
    ProofError,
    EncryptionError,
    DecryptionError,
    SigningError,
    SignatureVerificationError,
    SharedKeyNotFound,
}

impl fmt::Display for QuantumCryptoError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::EncapsulationError => write!(f, "Failed to encapsulate key"),
            Self::DecapsulationError => write!(f, "Failed to decapsulate key"),
            Self::KeyGenerationError => write!(f, "Failed to generate quantum-resistant keys"),
            Self::ProofError => write!(f, "Failed to generate or verify proof"),
            Self::EncryptionError => write!(f, "Failed to encrypt data"),
            Self::DecryptionError => write!(f, "Failed to decrypt data"),
            Self::SigningError => write!(f, "Failed to sign data"),
            Self::SignatureVerificationError => write!(f, "Failed to verify signature"),
            Self::SharedKeyNotFound => write!(f, "Shared key not found"),
        }
    }
}

impl StdError for QuantumCryptoError {}

/// Represents a zero-knowledge proof for transaction validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumProof {
    pub hash: String,
    pub signature: Vec<u8>,
    pub timestamp: u64,
}



pub trait HasTransactionData {
    fn get_sender(&self) -> &str;
    fn get_receiver(&self) -> &str;
    fn get_amount(&self) -> f64;
}

impl QuantumCrypto {
    /// Creates a new instance with generated key pairs
    pub fn new() -> Result<Self, QuantumCryptoError> {
        let mut rng = thread_rng();
        
        // Generate the ML-KEM key pair (using 768 for NIST security level 3)
        let (dk, ek) = MlKem768::generate(&mut rng);
        
        // Generate Dilithium keypair for signatures
        let (public_key, secret_key) = keypair();
        
        Ok(Self {
            decapsulation_key: Arc::new(dk),
            encapsulation_key: Arc::new(ek),
            signing_secret_key: secret_key,
            signing_public_key: public_key,
            shared_keys: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Encapsulates a shared secret using the public encapsulation key
    pub fn encapsulate(&self) -> Result<(Vec<u8>, SharedKey<Kem<MlKem768Params>>), QuantumCryptoError> {
        let mut rng = thread_rng();
        
        (*self.encapsulation_key)
            .encapsulate(&mut rng)
            .map(|(ct, shared)| (ct.to_vec(), shared))
            .map_err(|_| QuantumCryptoError::EncapsulationError)
    }

    /// Decapsulates a received ciphertext to recover the shared secret
    pub fn decapsulate(&self, ciphertext: &[u8]) -> Result<SharedKey<Kem<MlKem768Params>>, QuantumCryptoError> {
        // Convert slice to fixed-size Array
        let ct_array = Array::try_from(ciphertext)
            .map_err(|_| QuantumCryptoError::DecapsulationError)?;

        (*self.decapsulation_key)
            .decapsulate(&ct_array)
            .map_err(|_| QuantumCryptoError::DecapsulationError)
    }

    /// Returns a clone of the public encapsulation key that can be shared
    pub fn public_key(&self) -> Arc<EncapsulationKey<MlKem768Params>> {
        Arc::clone(&self.encapsulation_key)
    }

    // Generate a quantum-resistant proof for a transaction
    pub fn generate_proof(&self, data: &[u8]) -> Result<QuantumProof, QuantumCryptoError> {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let hash = format!("{:x}", hasher.finalize());

        // In a real implementation, this would use quantum-resistant signatures
        let signature = data.to_vec(); // Placeholder
        
        Ok(QuantumProof {
            hash,
            signature,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        })
    }

    // Verify a quantum-resistant proof
    pub fn verify_proof(&self, proof: &QuantumProof, data: &[u8]) -> Result<bool, QuantumCryptoError> {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let calculated_hash = format!("{:x}", hasher.finalize());

        Ok(calculated_hash == proof.hash)
    }

    /// Generates a geometric proof for a transaction with position data
    pub fn generate_geometric_proof(&self, transaction: &[u8], position: &Point) -> Result<GeometricProof, QuantumCryptoError> {
        let mut hasher = Sha256::new();
        hasher.update(transaction);
        hasher.update(&position.x.to_le_bytes());
        hasher.update(&position.y.to_le_bytes());
        let hash = format!("{:x}", hasher.finalize());

        // Use dilithium for quantum-resistant signatures
        let signature = detached_sign(transaction, &self.signing_secret_key)
            .as_bytes()
            .to_vec();
        let position_data = bincode::serialize(&position).map_err(|_| QuantumCryptoError::ProofError)?;
        
        Ok(GeometricProof {
            hash,
            signature,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            position_data,
            tessellation_points: Vec::new(), // Initialize with empty vector or actual tessellation points
            verification_path: Vec::new(),   // Initialize with empty vector or actual verification path
        })
    }

    /// Verifies a geometric proof
    pub fn verify_geometric_proof(&self, proof: &GeometricProof, data: &[u8]) -> Result<bool, QuantumCryptoError> {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let calculated_hash = format!("{:x}", hasher.finalize());

        Ok(calculated_hash == proof.hash)
    }

    // Add method for encrypting transaction data
    pub fn encrypt_transaction(&mut self, peer_id: &str, data: &[u8]) -> Result<Vec<u8>, QuantumCryptoError> {
        let (ciphertext, nonce) = self.encrypt_with_nonce(peer_id, data)?;
        
        let mut encrypted_data = Vec::with_capacity(nonce.len() + ciphertext.len());
        encrypted_data.extend_from_slice(&nonce);
        encrypted_data.extend_from_slice(&ciphertext);
        
        Ok(encrypted_data)
    }

    // Add method for decrypting transaction data
    pub fn decrypt_transaction(&self, peer_id: &str, encrypted_data: &[u8]) -> Result<Vec<u8>, QuantumCryptoError> {
        let (nonce, ciphertext) = encrypted_data.split_at(12);
        self.decrypt_with_nonce(peer_id, ciphertext, nonce)
    }

    // Add method for signing transactions
    pub fn sign_transaction(&self, transaction_data: &[u8]) -> Result<Vec<u8>, QuantumCryptoError> {
        let signature = detached_sign(transaction_data, &self.signing_secret_key);
        Ok(signature.as_bytes().to_vec())
    }

    // Add method for verifying transaction signatures
    pub fn verify_signature(&self, data: &[u8], signature: &[u8], public_key: &PublicKey) 
        -> Result<bool, QuantumCryptoError> {
        // Create DetachedSignature from the raw bytes
        let sig = match pqcrypto_dilithium::dilithium3::DetachedSignature::from_bytes(signature) {
            Ok(s) => s,
            Err(_) => return Ok(false)
        };
        
        verify_detached_signature(&sig, data, public_key)
            .map(|_| true)
            .map_err(|_| QuantumCryptoError::SignatureVerificationError)
    }

    // Helper method to get or establish shared key
    fn get_or_establish_shared_key(&mut self, peer_id: &str) -> Result<SharedKey<Kem<MlKem768Params>>, QuantumCryptoError> {
        // Check cache first
        if let Some((key, timestamp)) = self.shared_keys.lock().unwrap().get(peer_id) {
            // Check if key is still valid (e.g., less than 24 hours old)
            let age = SystemTime::now()
                .duration_since(*timestamp)
                .unwrap_or_default();
                
            if age.as_secs() < 24 * 60 * 60 {
                return Ok(key.clone());
            }
        }
        
        // Establish new shared key
        let (_ciphertext, shared_key) = self.encapsulate()?;
        
        // Store in cache
        self.shared_keys.lock().unwrap().insert(
            peer_id.to_string(),
            (shared_key.clone(), SystemTime::now())
        );
        
        Ok(shared_key)
    }

    // Helper method to get existing shared key
    fn get_shared_key(&self, peer_id: &str) -> Result<SharedKey<Kem<MlKem768Params>>, QuantumCryptoError> {
        self.shared_keys.lock().unwrap().get(peer_id)
            .map(|(key, _)| key.clone())
            .ok_or(QuantumCryptoError::SharedKeyNotFound)
    }

    // Add these helper methods
    fn encrypt_with_nonce(&mut self, peer_id: &str, data: &[u8]) 
        -> Result<(Vec<u8>, Vec<u8>), QuantumCryptoError> {
        // Get or establish shared key
        let shared_key = self.get_or_establish_shared_key(peer_id)?;
        
        // Create cipher instance using shared key
        let cipher = ChaCha20Poly1305::new(Key::from_slice(shared_key.as_ref()));
        
        // Generate random nonce using OsRng
        let nonce = ChaCha20Poly1305::generate_nonce(&mut OsRng);
        
        // Encrypt data
        let ciphertext = cipher.encrypt(&nonce, data)
            .map_err(|_| QuantumCryptoError::EncryptionError)?;
            
        Ok((ciphertext, nonce.to_vec()))
    }

    fn decrypt_with_nonce(&self, peer_id: &str, ciphertext: &[u8], nonce_bytes: &[u8]) 
        -> Result<Vec<u8>, QuantumCryptoError> {
        // Get shared key
        let shared_key = self.get_shared_key(peer_id)?;
        
        // Create cipher instance
        let cipher = ChaCha20Poly1305::new(Key::from_slice(shared_key.as_ref()));
        
        // Convert nonce bytes to Nonce type
        let nonce = Nonce::from_slice(nonce_bytes);
        
        // Decrypt data
        cipher.decrypt(nonce, ciphertext)
            .map_err(|_| QuantumCryptoError::DecryptionError)
    }

    // Add method to get public key
    pub fn get_signing_public_key(&self) -> &PublicKey {
        &self.signing_public_key
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_generation_and_encapsulation() {
        // Create a quantum crypto instance
        let alice = QuantumCrypto::new().unwrap();
        
        // Generate a separate encapsulation key for testing
        let mut rng = thread_rng();
        let (dk, ek) = MlKem768::generate(&mut rng);

        // Encapsulate a shared key
        let (ct, k_send) = ek.encapsulate(&mut rng).unwrap();
        let k_recv = dk.decapsulate(&ct).unwrap();
        assert_eq!(k_send, k_recv);
    }

    #[test]
    fn test_proof_generation_and_verification() {
        let crypto = QuantumCrypto::new().unwrap();
        let data = b"test data";
        let proof = crypto.generate_proof(data).unwrap();
        assert!(crypto.verify_proof(&proof, data).unwrap());
    }
}
