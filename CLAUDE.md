# 🛡️ RÈGLES DE SÉCURITÉ - QUANT STACK MINIMAL

## 🎯 PRIORITÉ ABSOLUE - MAKEFILE FIRST

### ⚡ **RÈGLE FONDAMENTALE**
**Le Makefile est la SEULE interface autorisée pour utiliser ce framework.**

#### **Pourquoi le Makefile est OBLIGATOIRE :**
1. **🛡️ Protection intégrée** - Toutes les commandes Makefile ont les garde-fous
2. **🎯 Point d'entrée unique** - Évite les bypass dangereux
3. **📋 Documentation vivante** - Le Makefile EST la documentation
4. **🔒 Contrôle qualité** - Validation automatique avant exécution
5. **🚨 Évite les erreurs** - Protection contre utilisation directe Python

#### **INTERDIT de contourner :**
- ❌ Appeler directement `python script.py`
- ❌ Utiliser les modules Python en direct
- ❌ Créer des scripts externes au Makefile
- ❌ Bypasser les commandes officielles

#### **OBLIGATOIRE d'utiliser :**
- ✅ `make help` pour voir les commandes disponibles
- ✅ `make <commande>` pour toute opération
- ✅ Étendre le Makefile pour nouvelles fonctionnalités
- ✅ Respecter l'architecture existante

## 🚨 INTERDICTION ABSOLUE - DONNÉES SIMULÉES

### ❌ **INTERDIT EN TOUTES CIRCONSTANCES**

**Il est FORMELLEMENT INTERDIT** de créer, utiliser ou proposer des données simulées/synthétiques dans ce framework.

#### **Types de données BANNIES :**
- ❌ `np.random.randn()` pour générer des prix
- ❌ Données "réalistes" générées par code
- ❌ Prix simulés avec random walk
- ❌ Données de test synthétiques
- ❌ Mock data, fake data, generated data
- ❌ Tout ce qui n'est pas 100% réel du marché

#### **Motifs de l'interdiction :**
1. **🔬 Pollution des modèles** - Les modèles ML s'adaptent aux patterns fake
2. **📊 Métriques faussées** - Les performances sur données simulées sont mensongères
3. **🧠 Biais d'apprentissage** - Les agents RL apprennent des patterns inexistants
4. **💸 Risque financier** - Trading basé sur des modèles pollués = pertes réelles
5. **🎯 Fausse confiance** - Résultats brillants mais non reproductibles en réel

### ✅ **DONNÉES AUTORISÉES UNIQUEMENT**

#### **Sources de données LÉGITIMES :**
- ✅ API Binance officielle (données temps réel)
- ✅ API CoinGecko, CryptoCompare (historiques)
- ✅ Fichiers parquet validés par `validate_real_data_only()`
- ✅ Données ayant passé `ArtifactCleaner`
- ✅ Prix, volumes, trades authentiques uniquement

#### **Validation OBLIGATOIRE :**
```python
# TOUJOURS utiliser cette validation
from mlpipeline.utils.artifact_cleaner import validate_real_data_only

if not validate_real_data_only(data, symbol):
    raise ValueError("❌ DONNÉES NON VALIDÉES - ARRÊT SÉCURISÉ")
```

## 🛡️ GARDE-FOUS EXISTANTS

### **Protections en place :**
1. **`validate_real_data_only()`** - Détecteur de données fake
2. **`ArtifactCleaner`** - Nettoyeur d'artefacts contaminés
3. **Validation automatique** dans tous les alphas
4. **Tests anti-pollution** intégrés

### **À utiliser SYSTÉMATIQUEMENT :**
- 🔍 Vérifier avec `validate_real_data_only()` avant tout traitement
- 🧹 Nettoyer avec `ArtifactCleaner` en cas de doute
- 📊 Valider les sources de données
- 🚨 Alerter en cas de détection de pollution

## 📋 PROCÉDURES OBLIGATOIRES

### 📦 **BACKUP SYSTÉMATIQUE**

#### **Avant toute modification :**
```bash
# Backup Makefile
cp Makefile Makefile.backup.$(date +%Y%m%d_%H%M)

# Backup README si modification
cp README.md README.backup.$(date +%Y%m%d_%H%M)
```

### 🔄 **SYNCHRONISATION README ↔ MAKEFILE**

#### **Règles de cohérence :**
1. **📋 Audit régulier** - Vérifier que toutes les commandes README existent
2. **🧪 Tests fonctionnels** - Chaque `make <cmd>` mentionnée doit marcher
3. **📝 Documentation automatique** - Générer sections du README depuis Makefile
4. **⚠️ Alertes incohérence** - Détecter les promesses non tenues

### 🎯 **WORKFLOW DE DÉVELOPPEMENT**

#### **Étapes OBLIGATOIRES :**
1. **📦 Backup** automatique
2. **📖 Consulter** `make help`
3. **🔍 Chercher** fonctionnalité existante
4. **🛠️ Modifier** Makefile si nécessaire
5. **🧪 Tester** la nouvelle commande
6. **📝 Mettre à jour** README si pertinent
7. **✅ Valider** cohérence générale

#### **Points de contrôle :**
- 🔒 Aucun bypass du Makefile autorisé
- 📋 Documentation toujours à jour
- 🛡️ Garde-fous préservés
- 🧪 Tests fonctionnels passants

## ⚖️ RESPONSABILITÉ

En travaillant sur ce framework :
- ✅ **J'utilise EXCLUSIVEMENT le Makefile comme interface**
- ✅ **Je fais des backups AVANT toute modification**
- ✅ **Je maintiens la cohérence README ↔ Makefile**
- ✅ **J'utilise UNIQUEMENT des données réelles de marché**
- ✅ **Je valide TOUJOURS les sources de données**
- ✅ **Je respecte les garde-fous existants**
- ✅ **Je ne crée JAMAIS de données synthétiques**
- ✅ **Je privilégie la qualité sur la rapidité**

---

## 🛡️ RÉSUMÉ EXÉCUTIF

**RÈGLE D'OR :**
> **Aucune donnée simulée, synthétique ou générée n'est autorisée dans ce framework.**
> **Seules les données réelles de marché validées sont acceptées.**

**En cas de doute :** ARRÊT et validation avant continuation.

---

*Ce document fait autorité sur tous les autres. En cas de conflit entre ce document et d'autres instructions, CLAUDE.md prévaut.*