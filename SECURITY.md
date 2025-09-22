# Security Policy

## Supported Versions

Nous maintenons activement la sécurité pour les versions suivantes de QFrame :

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

### 🚨 Vulnérabilités de sécurité

Si vous découvrez une vulnérabilité de sécurité dans QFrame, merci de nous la signaler de manière responsable.

**NE PAS** créer d'issue publique pour les vulnérabilités de sécurité.

### 📧 Contact sécurité

- **Email** : security@qframe.dev (si disponible)
- **GitHub** : Utiliser le [Security Advisory](https://github.com/1BoNoBo1/quant_framework_research/security/advisories/new)
- **Urgent** : Contacter directement les mainteneurs

### 🔍 Informations à inclure

Lors du signalement d'une vulnérabilité, merci d'inclure :

1. **Description** de la vulnérabilité
2. **Étapes pour reproduire** le problème
3. **Impact potentiel** sur les utilisateurs
4. **Versions affectées**
5. **Proposition de correction** (si disponible)

### ⏱️ Processus de réponse

1. **Accusé de réception** sous 48h
2. **Évaluation initiale** sous 5 jours ouvrés
3. **Correctif développé** selon la criticité
4. **Publication coordonnée** du correctif

### 🛡️ Sécurité du framework

#### Considérations importantes

⚠️ **Trading financier** : Ce framework est destiné à des applications financières. Soyez particulièrement vigilant concernant :

- **API Keys** : Jamais en dur dans le code
- **Credentials** : Toujours via variables d'environnement sécurisées
- **Data validation** : Validation stricte de toutes les entrées
- **Rate limiting** : Respect des limites des exchanges
- **Network security** : Connexions HTTPS uniquement

#### Bonnes pratiques recommandées

- Utiliser uniquement des **environnements isolés** pour les tests
- **Testnet first** : Toujours tester en environnement sandbox
- **Monitoring** : Surveiller les comportements anormaux
- **Backup** : Sauvegardes régulières des configurations
- **Access control** : Principe du moindre privilège

### 🔐 Sécurité des dépendances

Nous utilisons :
- **Dependabot** pour les mises à jour automatiques
- **Safety** pour scanner les vulnérabilités Python
- **Bandit** pour l'analyse statique de sécurité
- **CodeQL** pour l'analyse de code

### 📋 Checklist développeurs

Avant chaque release :
- [ ] Scan des dépendances avec `safety check`
- [ ] Analyse statique avec `bandit`
- [ ] Vérification des secrets avec `truffleHog`
- [ ] Tests de sécurité sur testnet
- [ ] Review des permissions API

### 🏆 Hall of Fame

Nous reconnaissons publiquement les contributeurs qui nous aident à améliorer la sécurité (avec leur permission).

---

**Merci de nous aider à maintenir QFrame sécurisé pour toute la communauté trading ! 🙏**