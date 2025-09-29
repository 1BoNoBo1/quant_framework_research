# 📋 TODOLIST COMPLÈTE POUR RÉSOUDRE LE PROBLÈME DE COUVERTURE

## 🎯 **OBJECTIF** : Passer de 5% à 75% de couverture RÉELLE

### **PHASE 1 : DIAGNOSTIC & CIBLAGE** ⚡

1. **Identifier les 10 modules les plus importants à couvrir**
   - Analyser les modules avec le plus de lignes de code
   - Prioriser core/, domain/, infrastructure/, api/
   - Mesurer l'impact potentiel de chaque module

### **PHASE 2 : TESTS D'EXÉCUTION RÉELLE** 🔥

2. **Créer tests d'exécution réelle pour qframe.core.config**
   - Instancier FrameworkConfig avec vraies valeurs
   - Tester validation, sérialisation, chargement env
   - Faire fonctionner toutes les méthodes

3. **Créer tests d'exécution réelle pour qframe.core.container**
   - DI avec vraies classes, pas mocks
   - Résolution complète de dépendances
   - Lifecycles singleton/transient réels

4. **Créer tests d'exécution réelle pour qframe.domain.services**
   - PortfolioService avec vrais calculs
   - ExecutionService avec vraies opérations
   - BacktestingService avec vrais backtests

5. **Créer tests d'exécution réelle pour qframe.infrastructure.persistence**
   - MemoryRepositories avec vraies données
   - CRUD operations complètes
   - Async/await correctement

6. **Créer tests d'exécution réelle pour qframe.api.main et routers**
   - FastAPI avec vrais endpoints
   - Vraies requêtes HTTP via TestClient
   - Réponses complètes parsées

### **PHASE 3 : CORRECTION DES TESTS EXISTANTS** 🔧

7. **Fixer les tests async avec await correct**
   - Corriger toutes les coroutines non awaited
   - Ajouter @pytest.mark.asyncio partout nécessaire
   - Tester les vraies méthodes async

8. **Remplacer les mocks par des appels réels dans tests existants**
   - Analyser tests/api/, tests/core/, tests/domain/
   - Remplacer Mock() par vraies instances
   - Faire exécuter le code réel

### **PHASE 4 : WORKFLOWS COMPLETS** 🚀

9. **Créer workflows complets end-to-end**
   - Portfolio → Strategy → Signal → Order → Execution
   - Data → Features → ML Training → Prediction
   - Config → Container → Services → API

### **PHASE 5 : OPTIMISATION & MESURE** 📊

10. **Mesurer couverture réelle module par module**
    - `pytest --cov=qframe.core`
    - `pytest --cov=qframe.domain`
    - `pytest --cov=qframe.infrastructure`
    - Identifier les gaps précis

11. **Optimiser les tests les plus impactants**
    - Focus sur modules avec le + de lignes non couvertes
    - 1 test efficace = +5% couverture
    - Éviter multiplication de tests inefficaces

12. **Atteindre 50% puis 75% de couverture effective**
    - Milestone 50% d'abord
    - Puis push vers 75%
    - Mesure continue pour éviter régression

---

## ⚠️ **RÈGLES D'OR**

- ✅ **EXÉCUTER** le code, pas juste l'importer
- ✅ **APPELER** les méthodes avec vrais paramètres
- ✅ **INSTANCIER** les classes complètement
- ❌ **ÉVITER** les mocks sauf nécessité absolue
- ❌ **STOP** créer des tests d'import inutiles
- ❌ **JAMAIS** oublier await sur async methods