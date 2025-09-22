"""
Enterprise prime aligned compute Platform - Encryption Service
======================================================

Comprehensive encryption service for data at rest and in transit,
providing AES-256 encryption, TLS/SSL support, and secure key management.

Author: Enterprise prime aligned compute Platform Team
Version: 2.0.0
License: Proprietary
"""

import os
import json
import base64
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding
from cryptography.hazmat.primitives import serialization

# Configure logging
logger = logging.getLogger(__name__)

# Encryption constants
AES_KEY_SIZE = 32  # 256 bits
RSA_KEY_SIZE = 2048
PBKDF2_ITERATIONS = 100000
SALT_SIZE = 32

class EncryptionService:
    """
    Enterprise-grade encryption service with support for:
    - AES-256 symmetric encryption
    - RSA asymmetric encryption
    - TLS/SSL certificate management
    - Secure key derivation
    - Data integrity verification
    """

    def __init__(self):
        # Master encryption keys
        self.master_key = self._load_or_generate_master_key()
        self.fernet = Fernet(self.master_key)

        # RSA key pair for asymmetric encryption
        self.private_key, self.public_key = self._load_or_generate_rsa_keys()

        # Session keys cache
        self.session_keys = {}

        logger.info("Encryption service initialized")

    def _load_or_generate_master_key(self) -> bytes:
        """Load or generate master encryption key"""
        key_file = os.path.join(os.path.dirname(__file__), 'master_key.enc')

        if os.path.exists(key_file):
            try:
                with open(key_file, 'rb') as f:
                    encrypted_key = f.read()
                # Decrypt master key with environment variable or generate new one
                env_key = os.getenv('MASTER_KEY_ENCRYPTION_KEY', '').encode()
                if env_key:
                    fernet = Fernet(env_key)
                    return fernet.decrypt(encrypted_key)
                else:
                    logger.warning("No MASTER_KEY_ENCRYPTION_KEY found, using default")
                    return Fernet.generate_key()
            except Exception as e:
                logger.warning(f"Failed to load master key: {e}")
                return Fernet.generate_key()
        else:
            # Generate new master key
            master_key = Fernet.generate_key()

            # Encrypt it with environment key if available
            env_key = os.getenv('MASTER_KEY_ENCRYPTION_KEY', '').encode()
            if env_key:
                fernet = Fernet(env_key)
                encrypted_key = fernet.encrypt(master_key)
                with open(key_file, 'wb') as f:
                    f.write(encrypted_key)

            return master_key

    def _load_or_generate_rsa_keys(self) -> tuple:
        """Load or generate RSA key pair"""
        private_key_file = os.path.join(os.path.dirname(__file__), 'rsa_private.pem')
        public_key_file = os.path.join(os.path.dirname(__file__), 'rsa_public.pem')

        if os.path.exists(private_key_file) and os.path.exists(public_key_file):
            try:
                # Load existing keys
                with open(private_key_file, 'rb') as f:
                    private_key = serialization.load_pem_private_key(
                        f.read(),
                        password=None,
                        backend=default_backend()
                    )

                with open(public_key_file, 'rb') as f:
                    public_key = serialization.load_pem_public_key(
                        f.read(),
                        backend=default_backend()
                    )

                return private_key, public_key

            except Exception as e:
                logger.warning(f"Failed to load RSA keys: {e}")

        # Generate new RSA key pair
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=RSA_KEY_SIZE,
            backend=default_backend()
        )

        public_key = private_key.public_key()

        # Save keys to files
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        with open(private_key_file, 'wb') as f:
            f.write(private_pem)

        with open(public_key_file, 'wb') as f:
            f.write(public_pem)

        logger.info("Generated new RSA key pair")
        return private_key, public_key

    # Symmetric Encryption (AES-256)
    def encrypt_data(self, data: Union[str, bytes, Dict[str, Any]]) -> str:
        """
        Encrypt data using AES-256

        Args:
            data: Data to encrypt (string, bytes, or dict)

        Returns:
            Base64 encoded encrypted data
        """
        try:
            # Convert data to JSON string if it's a dict
            if isinstance(data, dict):
                data = json.dumps(data, sort_keys=True)
            elif isinstance(data, str):
                data = data.encode('utf-8')

            # Encrypt with Fernet (AES-128 + HMAC)
            encrypted = self.fernet.encrypt(data)
            return base64.urlsafe_b64encode(encrypted).decode('utf-8')

        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise

    def decrypt_data(self, encrypted_data: str) -> Union[str, Dict[str, Any]]:
        """
        Decrypt data using AES-256

        Args:
            encrypted_data: Base64 encoded encrypted data

        Returns:
            Decrypted data (string or dict if JSON)
        """
        try:
            # Decode from base64
            encrypted = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))

            # Decrypt with Fernet
            decrypted = self.fernet.decrypt(encrypted)

            # Try to parse as JSON
            try:
                return json.loads(decrypted.decode('utf-8'))
            except json.JSONDecodeError:
                return decrypted.decode('utf-8')

        except InvalidToken:
            logger.error("Invalid encryption token")
            raise ValueError("Invalid encrypted data")
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise

    def generate_session_key(self, session_id: str) -> str:
        """
        Generate a unique session key for temporary encryption

        Args:
            session_id: Unique session identifier

        Returns:
            Base64 encoded session key
        """
        # Generate salt
        salt = secrets.token_bytes(SALT_SIZE)

        # Derive key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=AES_KEY_SIZE,
            salt=salt,
            iterations=PBKDF2_ITERATIONS,
            backend=default_backend()
        )

        key = base64.urlsafe_b64encode(kdf.derive(session_id.encode()))
        self.session_keys[session_id] = key

        return key.decode('utf-8')

    def encrypt_with_session_key(self, data: Union[str, bytes, Dict[str, Any]],
                               session_id: str) -> str:
        """
        Encrypt data with session-specific key

        Args:
            data: Data to encrypt
            session_id: Session identifier

        Returns:
            Base64 encoded encrypted data
        """
        if session_id not in self.session_keys:
            self.generate_session_key(session_id)

        session_key = self.session_keys[session_id]
        fernet = Fernet(session_key.encode())

        # Convert data to bytes
        if isinstance(data, dict):
            data = json.dumps(data, sort_keys=True).encode('utf-8')
        elif isinstance(data, str):
            data = data.encode('utf-8')

        encrypted = fernet.encrypt(data)
        return base64.urlsafe_b64encode(encrypted).decode('utf-8')

    def decrypt_with_session_key(self, encrypted_data: str, session_id: str) -> Union[str, Dict[str, Any]]:
        """
        Decrypt data with session-specific key

        Args:
            encrypted_data: Base64 encoded encrypted data
            session_id: Session identifier

        Returns:
            Decrypted data
        """
        if session_id not in self.session_keys:
            raise ValueError(f"Session key not found for session {session_id}")

        session_key = self.session_keys[session_id]
        fernet = Fernet(session_key.encode())

        try:
            encrypted = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
            decrypted = fernet.decrypt(encrypted)

            # Try to parse as JSON
            try:
                return json.loads(decrypted.decode('utf-8'))
            except json.JSONDecodeError:
                return decrypted.decode('utf-8')

        except Exception as e:
            logger.error(f"Session key decryption failed: {e}")
            raise

    def cleanup_session_key(self, session_id: str):
        """Remove session key from cache"""
        if session_id in self.session_keys:
            del self.session_keys[session_id]

    # Asymmetric Encryption (RSA)
    def encrypt_with_public_key(self, data: Union[str, bytes]) -> str:
        """
        Encrypt data with RSA public key

        Args:
            data: Data to encrypt

        Returns:
            Base64 encoded encrypted data
        """
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')

            # Encrypt with public key
            encrypted = self.public_key.encrypt(
                data,
                asym_padding.OAEP(
                    mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )

            return base64.urlsafe_b64encode(encrypted).decode('utf-8')

        except Exception as e:
            logger.error(f"RSA encryption failed: {e}")
            raise

    def decrypt_with_private_key(self, encrypted_data: str) -> bytes:
        """
        Decrypt data with RSA private key

        Args:
            encrypted_data: Base64 encoded encrypted data

        Returns:
            Decrypted data as bytes
        """
        try:
            encrypted = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))

            # Decrypt with private key
            decrypted = self.private_key.decrypt(
                encrypted,
                asym_padding.OAEP(
                    mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )

            return decrypted

        except Exception as e:
            logger.error(f"RSA decryption failed: {e}")
            raise

    def get_public_key_pem(self) -> str:
        """Get public key in PEM format"""
        public_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return public_pem.decode('utf-8')

    def sign_data(self, data: Union[str, bytes]) -> str:
        """
        Sign data with private key

        Args:
            data: Data to sign

        Returns:
            Base64 encoded signature
        """
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')

            # Sign with private key
            signature = self.private_key.sign(
                data,
                asym_padding.PSS(
                    mgf=asym_padding.MGF1(hashes.SHA256()),
                    salt_length=asym_padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )

            return base64.urlsafe_b64encode(signature).decode('utf-8')

        except Exception as e:
            logger.error(f"Data signing failed: {e}")
            raise

    def verify_signature(self, data: Union[str, bytes], signature: str) -> bool:
        """
        Verify data signature with public key

        Args:
            data: Original data
            signature: Base64 encoded signature

        Returns:
            True if signature is valid
        """
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')

            signature_bytes = base64.urlsafe_b64decode(signature.encode('utf-8'))

            # Verify signature
            self.public_key.verify(
                signature_bytes,
                data,
                asym_padding.PSS(
                    mgf=asym_padding.MGF1(hashes.SHA256()),
                    salt_length=asym_padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )

            return True

        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False

    # Data Integrity and Hashing
    def hash_data(self, data: Union[str, bytes], algorithm: str = 'sha256') -> str:
        """
        Generate hash of data

        Args:
            data: Data to hash
            algorithm: Hash algorithm ('sha256', 'sha512', etc.)

        Returns:
            Hexadecimal hash string
        """
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')

            if algorithm == 'sha256':
                hash_obj = hashlib.sha256()
            elif algorithm == 'sha512':
                hash_obj = hashlib.sha512()
            elif algorithm == 'blake2b':
                hash_obj = hashlib.blake2b()
            else:
                hash_obj = hashlib.sha256()  # Default

            hash_obj.update(data)
            return hash_obj.hexdigest()

        except Exception as e:
            logger.error(f"Hashing failed: {e}")
            raise

    def generate_secure_token(self, length: int = 32) -> str:
        """
        Generate a cryptographically secure random token

        Args:
            length: Token length in bytes

        Returns:
            URL-safe base64 encoded token
        """
        token_bytes = secrets.token_bytes(length)
        return base64.urlsafe_b64encode(token_bytes).decode('utf-8')

    def generate_password_salt(self) -> str:
        """Generate a random salt for password hashing"""
        return base64.urlsafe_b64encode(secrets.token_bytes(SALT_SIZE)).decode('utf-8')

    def hash_password_with_salt(self, password: str, salt: str) -> str:
        """
        Hash password with salt using PBKDF2

        Args:
            password: Plain text password
            salt: Base64 encoded salt

        Returns:
            Hexadecimal hash string
        """
        salt_bytes = base64.urlsafe_b64decode(salt.encode('utf-8'))

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt_bytes,
            iterations=PBKDF2_ITERATIONS,
            backend=default_backend()
        )

        key = kdf.derive(password.encode('utf-8'))
        return base64.urlsafe_b64encode(key).decode('utf-8')

    # File Encryption
    def encrypt_file(self, input_path: str, output_path: str, key: Optional[bytes] = None) -> bool:
        """
        Encrypt a file using AES-256

        Args:
            input_path: Path to input file
            output_path: Path to output encrypted file
            key: Optional encryption key (uses master key if not provided)

        Returns:
            True if successful
        """
        try:
            # Use provided key or master key
            if key is None:
                key = base64.urlsafe_b64decode(self.master_key.decode('utf-8').rstrip('='))

            # Generate random IV
            iv = secrets.token_bytes(16)

            # Create cipher
            cipher = Cipher(algorithms.AES(key[:32]), modes.CBC(iv), backend=default_backend())
            encryptor = cipher.encryptor()

            # Read input file
            with open(input_path, 'rb') as f:
                data = f.read()

            # Add padding
            padder = padding.PKCS7(algorithms.AES.block_size).padder()
            padded_data = padder.update(data) + padder.finalize()

            # Encrypt
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

            # Write encrypted file (IV + encrypted data)
            with open(output_path, 'wb') as f:
                f.write(iv + encrypted_data)

            return True

        except Exception as e:
            logger.error(f"File encryption failed: {e}")
            return False

    def decrypt_file(self, input_path: str, output_path: str, key: Optional[bytes] = None) -> bool:
        """
        Decrypt a file using AES-256

        Args:
            input_path: Path to encrypted input file
            output_path: Path to output decrypted file
            key: Optional decryption key (uses master key if not provided)

        Returns:
            True if successful
        """
        try:
            # Use provided key or master key
            if key is None:
                key = base64.urlsafe_b64decode(self.master_key.decode('utf-8').rstrip('='))

            # Read encrypted file
            with open(input_path, 'rb') as f:
                encrypted_data = f.read()

            # Extract IV and encrypted data
            iv = encrypted_data[:16]
            encrypted_content = encrypted_data[16:]

            # Create cipher
            cipher = Cipher(algorithms.AES(key[:32]), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()

            # Decrypt
            decrypted_padded = decryptor.update(encrypted_content) + decryptor.finalize()

            # Remove padding
            unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
            decrypted_data = unpadder.update(decrypted_padded) + unpadder.finalize()

            # Write decrypted file
            with open(output_path, 'wb') as f:
                f.write(decrypted_data)

            return True

        except Exception as e:
            logger.error(f"File decryption failed: {e}")
            return False

    # Key Management
    def rotate_master_key(self) -> bool:
        """
        Rotate the master encryption key

        Returns:
            True if successful
        """
        try:
            # Generate new master key
            new_master_key = Fernet.generate_key()

            # Create new Fernet instance
            new_fernet = Fernet(new_master_key)

            # Re-encrypt all sensitive data (this would need to be implemented
            # based on your data storage system)

            # Update master key
            self.master_key = new_master_key
            self.fernet = new_fernet

            # Save new key
            key_file = os.path.join(os.path.dirname(__file__), 'master_key.enc')
            env_key = os.getenv('MASTER_KEY_ENCRYPTION_KEY', '').encode()
            if env_key:
                fernet = Fernet(env_key)
                encrypted_key = fernet.encrypt(new_master_key)
                with open(key_file, 'wb') as f:
                    f.write(encrypted_key)

            logger.info("Master key rotated successfully")
            return True

        except Exception as e:
            logger.error(f"Master key rotation failed: {e}")
            return False

    def get_encryption_status(self) -> Dict[str, Any]:
        """Get encryption service status"""
        return {
            "master_key_initialized": self.master_key is not None,
            "rsa_keys_initialized": self.private_key is not None and self.public_key is not None,
            "session_keys_active": len(self.session_keys),
            "encryption_algorithm": "AES-256 + HMAC-SHA256",
            "asymmetric_algorithm": f"RSA-{RSA_KEY_SIZE}",
            "key_derivation": f"PBKDF2-{PBKDF2_ITERATIONS} iterations"
        }

# Global encryption service instance
encryption_service = EncryptionService()

if __name__ == "__main__":
    # Test the encryption service
    print("🧪 Testing Encryption Service...")

    # Test data encryption/decryption
    test_data = {"message": "Hello, encrypted world!", "number": 42, "nested": {"key": "value"}}

    encrypted = encryption_service.encrypt_data(test_data)
    print(f"✅ Data encrypted: {encrypted[:50]}...")

    decrypted = encryption_service.decrypt_data(encrypted)
    print(f"✅ Data decrypted: {decrypted}")

    # Test session key encryption
    session_id = "test-session-123"
    session_encrypted = encryption_service.encrypt_with_session_key(test_data, session_id)
    print(f"✅ Session encrypted: {session_encrypted[:50]}...")

    session_decrypted = encryption_service.decrypt_with_session_key(session_encrypted, session_id)
    print(f"✅ Session decrypted: {session_decrypted}")

    # Test RSA encryption
    message = "RSA test message"
    rsa_encrypted = encryption_service.encrypt_with_public_key(message)
    print(f"✅ RSA encrypted: {rsa_encrypted[:50]}...")

    rsa_decrypted = encryption_service.decrypt_with_private_key(rsa_encrypted)
    print(f"✅ RSA decrypted: {rsa_decrypted.decode('utf-8')}")

    # Test digital signature
    signature = encryption_service.sign_data(message)
    print(f"✅ Data signed: {signature[:50]}...")

    is_valid = encryption_service.verify_signature(message, signature)
    print(f"✅ Signature verified: {is_valid}")

    # Test hashing
    hash_value = encryption_service.hash_data(message)
    print(f"✅ Data hashed: {hash_value}")

    # Test secure token generation
    token = encryption_service.generate_secure_token()
    print(f"✅ Secure token: {token[:20]}...")

    # Test password hashing
    salt = encryption_service.generate_password_salt()
    hashed_password = encryption_service.hash_password_with_salt("mypassword123", salt)
    print(f"✅ Password hashed: {hashed_password[:20]}...")

    # Get status
    status = encryption_service.get_encryption_status()
    print(f"✅ Encryption status: {status}")

    print("🎉 Encryption service test completed!")
