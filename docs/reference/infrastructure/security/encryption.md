# qframe.infrastructure.security.encryption


Security Infrastructure: Encryption & Secret Management
=======================================================

Système de chiffrement pour secrets sensibles (API keys, tokens, etc.).
Utilise Fernet (AES 128 CBC + HMAC SHA256) pour chiffrement symétrique sécurisé.


::: qframe.infrastructure.security.encryption
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
      members_order: alphabetical
      filters:
        - "!^_"
        - "!^__"
      group_by_category: true
      show_category_heading: true

## Composants

### Classes

- `EncryptionConfig`
- `SecretManager`
- `EncryptedField`

### Fonctions

- `get_secret_manager`
- `init_secret_manager`
- `encrypt_secret`
- `decrypt_secret`

