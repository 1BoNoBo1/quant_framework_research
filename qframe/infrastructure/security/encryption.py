"""
Security Infrastructure: Encryption & Secret Management
=======================================================

Système de chiffrement pour secrets sensibles (API keys, tokens, etc.).
Utilise Fernet (AES 128 CBC + HMAC SHA256) pour chiffrement symétrique sécurisé.
"""

import os
import base64
from typing import Optional, Union, Dict, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import secrets
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
import json

from qframe.infrastructure.observability.logging import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


@dataclass
class EncryptionConfig:
    """Configuration pour le système de chiffrement"""
    key_file: str = ".qframe_key"
    salt_file: str = ".qframe_salt"
    key_derivation_iterations: int = 100000  # PBKDF2 iterations
    use_env_key: bool = True  # Utiliser QFRAME_ENCRYPTION_KEY en priorité


class SecretManager:
    """
    Gestionnaire de secrets chiffrés pour QFrame.

    Fonctionnalités:
    - Chiffrement AES-256 avec Fernet
    - Dérivation de clé sécurisée (PBKDF2)
    - Rotation de clés
    - Audit des accès aux secrets
    """

    def __init__(self, config: Optional[EncryptionConfig] = None):
        self.config = config or EncryptionConfig()
        self._fernet: Optional[Fernet] = None
        self._master_key: Optional[bytes] = None
        self._initialized = False

    def initialize(self, master_password: Optional[str] = None) -> None:
        """
        Initialise le gestionnaire de secrets.

        Args:
            master_password: Mot de passe maître (optionnel si QFRAME_ENCRYPTION_KEY existe)
        """
        try:
            # 1. Tenter de récupérer la clé depuis l'environnement
            if self.config.use_env_key:
                env_key = os.getenv('QFRAME_ENCRYPTION_KEY')
                if env_key:
                    self._master_key = base64.b64decode(env_key.encode())
                    self._fernet = Fernet(base64.b64encode(self._master_key[:32]))
                    self._initialized = True
                    logger.info("Encryption initialized from environment variable")
                    return

            # 2. Générer ou récupérer la clé depuis les fichiers
            if master_password:
                self._master_key = self._derive_key_from_password(master_password)
            else:
                self._master_key = self._load_or_generate_key()

            # 3. Créer l'instance Fernet
            fernet_key = base64.b64encode(self._master_key[:32])  # Fernet needs 32 bytes
            self._fernet = Fernet(fernet_key)
            self._initialized = True

            logger.info("Secret manager initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize secret manager", error=e)
            raise

    def _derive_key_from_password(self, password: str) -> bytes:
        """Dérive une clé cryptographique depuis un mot de passe"""
        # Charger ou générer le salt
        salt = self._load_or_generate_salt()

        # Dériver la clé avec PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.config.key_derivation_iterations,
        )

        return kdf.derive(password.encode())

    def _load_or_generate_salt(self) -> bytes:
        """Charge ou génère le salt pour la dérivation de clé"""
        salt_path = Path(self.config.salt_file)

        if salt_path.exists():
            with open(salt_path, 'rb') as f:
                return f.read()
        else:
            # Générer un nouveau salt
            salt = os.urandom(16)  # 128 bits
            with open(salt_path, 'wb') as f:
                f.write(salt)

            # Sécuriser le fichier
            os.chmod(salt_path, 0o600)
            logger.info(f"Generated new salt: {salt_path}")
            return salt

    def _load_or_generate_key(self) -> bytes:
        """Charge ou génère une clé maître"""
        key_path = Path(self.config.key_file)

        if key_path.exists():
            with open(key_path, 'rb') as f:
                return f.read()
        else:
            # Générer une nouvelle clé
            key = Fernet.generate_key()
            with open(key_path, 'wb') as f:
                f.write(key)

            # Sécuriser le fichier
            os.chmod(key_path, 0o600)
            logger.info(f"Generated new encryption key: {key_path}")
            return key

    def encrypt(self, plaintext: str) -> str:
        """
        Chiffre un texte en clair.

        Args:
            plaintext: Texte à chiffrer

        Returns:
            Texte chiffré encodé en base64
        """
        if not self._initialized:
            raise RuntimeError("SecretManager not initialized")

        if not plaintext:
            return ""

        try:
            encrypted_data = self._fernet.encrypt(plaintext.encode())
            result = base64.b64encode(encrypted_data).decode()

            logger.audit("Secret encrypted",
                        secret_hash=hashlib.sha256(plaintext.encode()).hexdigest()[:8])
            return result

        except Exception as e:
            logger.error("Failed to encrypt secret", error=e)
            raise

    def decrypt(self, encrypted_text: str) -> str:
        """
        Déchiffre un texte chiffré.

        Args:
            encrypted_text: Texte chiffré en base64

        Returns:
            Texte en clair
        """
        if not self._initialized:
            raise RuntimeError("SecretManager not initialized")

        if not encrypted_text:
            return ""

        try:
            encrypted_data = base64.b64decode(encrypted_text.encode())
            decrypted_bytes = self._fernet.decrypt(encrypted_data)
            result = decrypted_bytes.decode()

            logger.audit("Secret decrypted",
                        secret_hash=hashlib.sha256(result.encode()).hexdigest()[:8])
            return result

        except Exception as e:
            logger.error("Failed to decrypt secret", error=e)
            raise

    def rotate_key(self, new_master_password: Optional[str] = None) -> None:
        """
        Effectue une rotation de la clé de chiffrement.

        Args:
            new_master_password: Nouveau mot de passe maître
        """
        if not self._initialized:
            raise RuntimeError("SecretManager not initialized")

        logger.info("Starting key rotation")

        # Générer une nouvelle clé
        if new_master_password:
            new_key = self._derive_key_from_password(new_master_password)
        else:
            new_key = Fernet.generate_key()

        # Sauvegarder l'ancienne configuration
        old_fernet = self._fernet

        # Créer la nouvelle instance Fernet
        fernet_key = base64.b64encode(new_key[:32])
        new_fernet = Fernet(fernet_key)

        # Mettre à jour
        self._master_key = new_key
        self._fernet = new_fernet

        # Sauvegarder la nouvelle clé
        key_path = Path(self.config.key_file)
        with open(key_path, 'wb') as f:
            f.write(new_key)

        logger.info("Key rotation completed successfully")

    def export_key_for_env(self) -> str:
        """
        Exporte la clé pour utilisation en variable d'environnement.

        Returns:
            Clé encodée en base64 pour QFRAME_ENCRYPTION_KEY
        """
        if not self._initialized:
            raise RuntimeError("SecretManager not initialized")

        env_key = base64.b64encode(self._master_key).decode()
        logger.info("Encryption key exported for environment variable")
        return env_key

    def is_initialized(self) -> bool:
        """Vérifie si le gestionnaire est initialisé"""
        return self._initialized


class EncryptedField:
    """
    Champ chiffré pour Pydantic models.

    Usage:
    ```python
    class Config(BaseModel):
        api_key: str = EncryptedField(default="")
    ```
    """

    def __init__(self, default: str = "", description: str = ""):
        self.default = default
        self.description = description
        self._secret_manager: Optional[SecretManager] = None

    def set_secret_manager(self, secret_manager: SecretManager):
        """Configure le gestionnaire de secrets"""
        self._secret_manager = secret_manager

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self

        # Récupérer la valeur chiffrée
        encrypted_value = getattr(obj, f"_{self.name}", self.default)

        if not encrypted_value or not self._secret_manager:
            return encrypted_value

        # Déchiffrer
        try:
            return self._secret_manager.decrypt(encrypted_value)
        except Exception:
            logger.warning(f"Failed to decrypt field {self.name}")
            return encrypted_value

    def __set__(self, obj, value):
        if not value:
            setattr(obj, f"_{self.name}", value)
            return

        # Chiffrer si possible
        if self._secret_manager:
            try:
                encrypted_value = self._secret_manager.encrypt(value)
                setattr(obj, f"_{self.name}", encrypted_value)
                return
            except Exception:
                logger.warning(f"Failed to encrypt field {self.name}")

        # Fallback: stocker en clair avec avertissement
        setattr(obj, f"_{self.name}", value)
        logger.warning(f"Field {self.name} stored in plaintext")

    def __set_name__(self, owner, name):
        self.name = name


# Instance globale du gestionnaire de secrets
_global_secret_manager: Optional[SecretManager] = None


def get_secret_manager() -> SecretManager:
    """Récupère l'instance globale du gestionnaire de secrets"""
    global _global_secret_manager

    if _global_secret_manager is None:
        _global_secret_manager = SecretManager()

        # Initialisation automatique
        try:
            _global_secret_manager.initialize()
        except Exception as e:
            logger.error("Failed to auto-initialize secret manager", error=e)

    return _global_secret_manager


def init_secret_manager(master_password: Optional[str] = None,
                       config: Optional[EncryptionConfig] = None) -> SecretManager:
    """
    Initialise le gestionnaire de secrets global.

    Args:
        master_password: Mot de passe maître
        config: Configuration personnalisée

    Returns:
        Instance du gestionnaire de secrets
    """
    global _global_secret_manager

    _global_secret_manager = SecretManager(config)
    _global_secret_manager.initialize(master_password)

    return _global_secret_manager


# Utilitaires pour usage direct
def encrypt_secret(plaintext: str) -> str:
    """Chiffre un secret avec le gestionnaire global"""
    return get_secret_manager().encrypt(plaintext)


def decrypt_secret(encrypted_text: str) -> str:
    """Déchiffre un secret avec le gestionnaire global"""
    return get_secret_manager().decrypt(encrypted_text)