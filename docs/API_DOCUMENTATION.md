# 📚 Documentation API QFrame

## Informations Générales

- **Titre**: QFrame API
- **Version**: 1.0.0
- **Description**: Backend API pour le framework de trading quantitatif QFrame
- **URL de base**: `http://localhost:8000`

## Authentification

L'API utilise un middleware d'authentification basique. Pour les tests de développement, aucune authentification n'est requise sur les endpoints publics.

## Endpoints Disponibles

### Market Data

#### GET `/api/v1/market-data/symbols`

**Résumé**: Get Supported Symbols

**Description**: Récupère la liste des symboles supportés.

**Paramètres**:
- `exchange` (string) (optionnel): Exchange spécifique

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/market-data/symbols
```

---

#### GET `/api/v1/market-data/price/{symbol}`

**Résumé**: Get Current Price

**Description**: Récupère le prix actuel d'un symbole.

**Paramètres**:
- `symbol` (string) (requis): 

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/market-data/price/{symbol}
```

---

#### GET `/api/v1/market-data/prices`

**Résumé**: Get Multiple Prices

**Description**: Récupère les prix de plusieurs symboles.

**Paramètres**:
- `symbols` (string) (requis): Symboles séparés par des virgules

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/market-data/prices
```

---

#### GET `/api/v1/market-data/ohlcv/{symbol}`

**Résumé**: Get Ohlcv Data

**Description**: Récupère les données OHLCV pour un symbole.

**Paramètres**:
- `symbol` (string) (requis): 
- `timeframe` (string) (optionnel): Timeframe
- `start_date` (string) (optionnel): Date de début
- `end_date` (string) (optionnel): Date de fin
- `limit` (integer) (optionnel): Nombre de bougies

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/market-data/ohlcv/{symbol}
```

---

#### GET `/api/v1/market-data/ticker/{symbol}`

**Résumé**: Get Ticker Data

**Description**: Récupère les données ticker complètes pour un symbole.

**Paramètres**:
- `symbol` (string) (requis): 

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/market-data/ticker/{symbol}
```

---

#### GET `/api/v1/market-data/depth/{symbol}`

**Résumé**: Get Order Book

**Description**: Récupère le carnet d'ordres pour un symbole.

**Paramètres**:
- `symbol` (string) (requis): 
- `limit` (integer) (optionnel): Nombre de niveaux

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/market-data/depth/{symbol}
```

---

#### GET `/api/v1/market-data/trades/{symbol}`

**Résumé**: Get Recent Trades

**Description**: Récupère les trades récents pour un symbole.

**Paramètres**:
- `symbol` (string) (requis): 
- `limit` (integer) (optionnel): Nombre de trades

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/market-data/trades/{symbol}
```

---

#### GET `/api/v1/market-data/exchanges`

**Résumé**: Get Supported Exchanges

**Description**: Récupère la liste des exchanges supportés.

**Réponses**:
- `200`: Successful Response

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/market-data/exchanges
```

---

#### GET `/api/v1/market-data/status`

**Résumé**: Get Market Status

**Description**: Récupère le statut des marchés.

**Réponses**:
- `200`: Successful Response

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/market-data/status
```

---

#### POST `/api/v1/market-data/bulk`

**Résumé**: Get Bulk Market Data

**Description**: Récupère des données de marché en bulk.

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X POST http://localhost:8000/api/v1/market-data/bulk \\\n  -H "Content-Type: application/json" \\\n  -d '{"example": "data"}'
```

---

#### GET `/api/v1/market-data/stats/{symbol}`

**Résumé**: Get Market Stats

**Description**: Récupère les statistiques de marché pour un symbole.

**Paramètres**:
- `symbol` (string) (requis): 
- `period` (string) (optionnel): Période pour les stats

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/market-data/stats/{symbol}
```

---

### Orders

#### POST `/api/v1/orders/`

**Résumé**: Create Order

**Description**: Crée un nouvel ordre.

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X POST http://localhost:8000/api/v1/orders/ \\\n  -H "Content-Type: application/json" \\\n  -d '{"example": "data"}'
```

---

#### GET `/api/v1/orders/`

**Résumé**: Get Orders

**Description**: Récupère la liste des ordres avec pagination et filtres.

**Paramètres**:
- `page` (integer) (optionnel): Numéro de page
- `per_page` (integer) (optionnel): Ordres par page
- `symbol` (string) (optionnel): Filtrer par symbole
- `status` (string) (optionnel): Filtrer par statut
- `side` (string) (optionnel): Filtrer par côté
- `order_type` (string) (optionnel): Filtrer par type
- `start_date` (string) (optionnel): Date de début
- `end_date` (string) (optionnel): Date de fin

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/orders/
```

---

#### GET `/api/v1/orders/{order_id}`

**Résumé**: Get Order

**Description**: Récupère un ordre spécifique.

**Paramètres**:
- `order_id` (string) (requis): ID de l'ordre

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/orders/{order_id}
```

---

#### PUT `/api/v1/orders/{order_id}`

**Résumé**: Update Order

**Description**: Met à jour un ordre existant.

**Paramètres**:
- `order_id` (string) (requis): ID de l'ordre

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X PUT http://localhost:8000/api/v1/orders/{order_id} \\\n  -H "Content-Type: application/json" \\\n  -d '{"example": "data"}'
```

---

#### DELETE `/api/v1/orders/{order_id}`

**Résumé**: Cancel Order

**Description**: Annule un ordre.

**Paramètres**:
- `order_id` (string) (requis): ID de l'ordre

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X DELETE http://localhost:8000/api/v1/orders/{order_id}
```

---

#### POST `/api/v1/orders/bulk`

**Résumé**: Create Bulk Orders

**Description**: Crée plusieurs ordres en une seule requête.

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X POST http://localhost:8000/api/v1/orders/bulk \\\n  -H "Content-Type: application/json" \\\n  -d '{"example": "data"}'
```

---

#### GET `/api/v1/orders/active/count`

**Résumé**: Get Active Orders Count

**Description**: Récupère le nombre d'ordres actifs.

**Paramètres**:
- `symbol` (string) (optionnel): Symbole spécifique

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/orders/active/count
```

---

#### POST `/api/v1/orders/cancel-all`

**Résumé**: Cancel All Orders

**Description**: Annule tous les ordres actifs.

**Paramètres**:
- `symbol` (string) (optionnel): Symbole spécifique (sinon tous)
- `confirm` (boolean) (optionnel): Confirmation required

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X POST http://localhost:8000/api/v1/orders/cancel-all \\\n  -H "Content-Type: application/json" \\\n  -d '{"example": "data"}'
```

---

#### GET `/api/v1/orders/history/{order_id}`

**Résumé**: Get Order History

**Description**: Récupère l'historique des modifications d'un ordre.

**Paramètres**:
- `order_id` (string) (requis): ID de l'ordre

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/orders/history/{order_id}
```

---

#### GET `/api/v1/orders/fills/{order_id}`

**Résumé**: Get Order Fills

**Description**: Récupère les exécutions partielles d'un ordre.

**Paramètres**:
- `order_id` (string) (requis): ID de l'ordre

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/orders/fills/{order_id}
```

---

#### GET `/api/v1/orders/statistics`

**Résumé**: Get Order Statistics

**Description**: Récupère les statistiques des ordres.

**Paramètres**:
- `start_date` (string) (optionnel): Date de début
- `end_date` (string) (optionnel): Date de fin
- `symbol` (string) (optionnel): Symbole spécifique

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/orders/statistics
```

---

### Positions

#### GET `/api/v1/positions/`

**Résumé**: Get Positions

**Description**: Récupère la liste des positions avec pagination et filtres.

**Paramètres**:
- `page` (integer) (optionnel): Numéro de page
- `per_page` (integer) (optionnel): Positions par page
- `symbol` (string) (optionnel): Filtrer par symbole
- `side` (string) (optionnel): Filtrer par côté (LONG/SHORT)
- `status` (string) (optionnel): Filtrer par statut

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/positions/
```

---

#### GET `/api/v1/positions/{position_id}`

**Résumé**: Get Position

**Description**: Récupère une position spécifique.

**Paramètres**:
- `position_id` (string) (requis): ID de la position

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/positions/{position_id}
```

---

#### DELETE `/api/v1/positions/{position_id}`

**Résumé**: Close Position

**Description**: Ferme une position.

**Paramètres**:
- `position_id` (string) (requis): ID de la position
- `close_price` (string) (optionnel): Prix de fermeture (market si non spécifié)

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X DELETE http://localhost:8000/api/v1/positions/{position_id}
```

---

#### PUT `/api/v1/positions/{position_id}/stop-loss`

**Résumé**: Update Stop Loss

**Description**: Met à jour le stop loss d'une position.

**Paramètres**:
- `position_id` (string) (requis): ID de la position
- `stop_loss_price` (number) (requis): Nouveau prix de stop loss

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X PUT http://localhost:8000/api/v1/positions/{position_id}/stop-loss \\\n  -H "Content-Type: application/json" \\\n  -d '{"example": "data"}'
```

---

#### PUT `/api/v1/positions/{position_id}/take-profit`

**Résumé**: Update Take Profit

**Description**: Met à jour le take profit d'une position.

**Paramètres**:
- `position_id` (string) (requis): ID de la position
- `take_profit_price` (number) (requis): Nouveau prix de take profit

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X PUT http://localhost:8000/api/v1/positions/{position_id}/take-profit \\\n  -H "Content-Type: application/json" \\\n  -d '{"example": "data"}'
```

---

#### POST `/api/v1/positions/close-all`

**Résumé**: Close All Positions

**Description**: Ferme toutes les positions.

**Paramètres**:
- `symbol` (string) (optionnel): Symbole spécifique (sinon toutes)
- `confirm` (boolean) (optionnel): Confirmation required

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X POST http://localhost:8000/api/v1/positions/close-all \\\n  -H "Content-Type: application/json" \\\n  -d '{"example": "data"}'
```

---

#### GET `/api/v1/positions/portfolio/summary`

**Résumé**: Get Portfolio Summary

**Description**: Récupère le résumé du portefeuille.

**Réponses**:
- `200`: Successful Response

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/positions/portfolio/summary
```

---

#### GET `/api/v1/positions/portfolio/allocation`

**Résumé**: Get Portfolio Allocation

**Description**: Récupère l'allocation du portefeuille par asset.

**Réponses**:
- `200`: Successful Response

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/positions/portfolio/allocation
```

---

#### GET `/api/v1/positions/portfolio/performance`

**Résumé**: Get Portfolio Performance

**Description**: Récupère les performances du portefeuille.

**Paramètres**:
- `start_date` (string) (optionnel): Date de début
- `end_date` (string) (optionnel): Date de fin

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/positions/portfolio/performance
```

---

#### GET `/api/v1/positions/analytics/pnl`

**Résumé**: Get Pnl Analytics

**Description**: Récupère l'analyse PnL détaillée.

**Paramètres**:
- `period` (string) (optionnel): Période d'analyse
- `symbol` (string) (optionnel): Symbole spécifique

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/positions/analytics/pnl
```

---

#### GET `/api/v1/positions/history/{position_id}`

**Résumé**: Get Position History

**Description**: Récupère l'historique d'une position.

**Paramètres**:
- `position_id` (string) (requis): ID de la position

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/positions/history/{position_id}
```

---

#### GET `/api/v1/positions/exposure/net`

**Résumé**: Get Net Exposure

**Description**: Récupère l'exposition nette.

**Paramètres**:
- `symbol` (string) (optionnel): Symbole spécifique

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/positions/exposure/net
```

---

#### GET `/api/v1/positions/statistics`

**Résumé**: Get Position Statistics

**Description**: Récupère les statistiques des positions.

**Paramètres**:
- `start_date` (string) (optionnel): Date de début
- `end_date` (string) (optionnel): Date de fin
- `symbol` (string) (optionnel): Symbole spécifique

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/positions/statistics
```

---

### Risk Management

#### GET `/api/v1/risk/metrics`

**Résumé**: Get Risk Metrics

**Description**: Récupère les métriques de risque actuelles.

**Réponses**:
- `200`: Successful Response

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/risk/metrics
```

---

#### GET `/api/v1/risk/var`

**Résumé**: Get Var Calculation

**Description**: Calcule la VaR du portefeuille.

**Paramètres**:
- `confidence_level` (number) (optionnel): Niveau de confiance
- `time_horizon` (integer) (optionnel): Horizon temporel en jours
- `method` (string) (optionnel): Méthode de calcul

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/risk/var
```

---

#### GET `/api/v1/risk/stress-test`

**Résumé**: Run Stress Test

**Description**: Exécute un stress test sur le portefeuille.

**Paramètres**:
- `scenario` (string) (requis): Scénario de stress test

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/risk/stress-test
```

---

#### GET `/api/v1/risk/limits`

**Résumé**: Get Risk Limits

**Description**: Récupère les limites de risque configurées.

**Réponses**:
- `200`: Successful Response

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/risk/limits
```

---

#### PUT `/api/v1/risk/limits`

**Résumé**: Update Risk Limits

**Description**: Met à jour les limites de risque.

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X PUT http://localhost:8000/api/v1/risk/limits \\\n  -H "Content-Type: application/json" \\\n  -d '{"example": "data"}'
```

---

#### GET `/api/v1/risk/alerts`

**Résumé**: Get Risk Alerts

**Description**: Récupère les alertes de risque.

**Paramètres**:
- `active_only` (boolean) (optionnel): Seulement les alertes actives
- `severity` (string) (optionnel): Niveau de sévérité

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/risk/alerts
```

---

#### POST `/api/v1/risk/alerts/{alert_id}/acknowledge`

**Résumé**: Acknowledge Alert

**Description**: Accuse réception d'une alerte de risque.

**Paramètres**:
- `alert_id` (string) (requis): 

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X POST http://localhost:8000/api/v1/risk/alerts/{alert_id}/acknowledge \\\n  -H "Content-Type: application/json" \\\n  -d '{"example": "data"}'
```

---

#### GET `/api/v1/risk/exposure`

**Résumé**: Get Risk Exposure

**Description**: Récupère l'exposition aux risques.

**Paramètres**:
- `by_asset` (boolean) (optionnel): Exposition par asset
- `by_sector` (boolean) (optionnel): Exposition par secteur

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/risk/exposure
```

---

#### GET `/api/v1/risk/correlation`

**Résumé**: Get Correlation Matrix

**Description**: Récupère la matrice de corrélation des positions.

**Paramètres**:
- `lookback_days` (integer) (optionnel): Période de lookback en jours

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/risk/correlation
```

---

#### GET `/api/v1/risk/concentration`

**Résumé**: Get Concentration Risk

**Description**: Récupère l'analyse du risque de concentration.

**Réponses**:
- `200`: Successful Response

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/risk/concentration
```

---

#### GET `/api/v1/risk/liquidity`

**Résumé**: Get Liquidity Risk

**Description**: Récupère l'analyse du risque de liquidité.

**Paramètres**:
- `time_horizon` (string) (optionnel): Horizon de liquidation

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/risk/liquidity
```

---

#### POST `/api/v1/risk/emergency-stop`

**Résumé**: Emergency Stop

**Description**: Déclenche un arrêt d'urgence du trading.

**Paramètres**:
- `reason` (string) (requis): Raison de l'arrêt d'urgence
- `confirm` (boolean) (optionnel): Confirmation required

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X POST http://localhost:8000/api/v1/risk/emergency-stop \\\n  -H "Content-Type: application/json" \\\n  -d '{"example": "data"}'
```

---

#### GET `/api/v1/risk/backtesting/risk`

**Résumé**: Get Backtesting Risk Metrics

**Description**: Récupère les métriques de risque d'un backtest.

**Paramètres**:
- `backtest_id` (string) (requis): ID du backtest

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/risk/backtesting/risk
```

---

#### GET `/api/v1/risk/scenario-analysis`

**Résumé**: Run Scenario Analysis

**Description**: Exécute une analyse de scénarios multiples.

**Paramètres**:
- `scenarios` (string) (requis): Scénarios séparés par des virgules

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/risk/scenario-analysis
```

---

### Strategies

#### GET `/api/v1/strategies/`

**Résumé**: Get Strategies

**Description**: Récupère la liste des stratégies avec pagination et filtres.

**Paramètres**:
- `page` (integer) (optionnel): Numéro de page
- `per_page` (integer) (optionnel): Stratégies par page
- `type` (string) (optionnel): Filtrer par type
- `status` (string) (optionnel): Filtrer par statut

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/strategies/
```

---

#### POST `/api/v1/strategies/`

**Résumé**: Create Strategy

**Description**: Crée une nouvelle stratégie.

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X POST http://localhost:8000/api/v1/strategies/ \\\n  -H "Content-Type: application/json" \\\n  -d '{"example": "data"}'
```

---

#### GET `/api/v1/strategies/{strategy_id}`

**Résumé**: Get Strategy

**Description**: Récupère une stratégie spécifique.

**Paramètres**:
- `strategy_id` (string) (requis): ID de la stratégie

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/strategies/{strategy_id}
```

---

#### PUT `/api/v1/strategies/{strategy_id}`

**Résumé**: Update Strategy

**Description**: Met à jour une stratégie existante.

**Paramètres**:
- `strategy_id` (string) (requis): ID de la stratégie

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X PUT http://localhost:8000/api/v1/strategies/{strategy_id} \\\n  -H "Content-Type: application/json" \\\n  -d '{"example": "data"}'
```

---

#### DELETE `/api/v1/strategies/{strategy_id}`

**Résumé**: Delete Strategy

**Description**: Supprime une stratégie.

**Paramètres**:
- `strategy_id` (string) (requis): ID de la stratégie
- `force` (boolean) (optionnel): Forcer la suppression même si active

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X DELETE http://localhost:8000/api/v1/strategies/{strategy_id}
```

---

#### POST `/api/v1/strategies/{strategy_id}/start`

**Résumé**: Start Strategy

**Description**: Démarre une stratégie.

**Paramètres**:
- `strategy_id` (string) (requis): ID de la stratégie

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X POST http://localhost:8000/api/v1/strategies/{strategy_id}/start \\\n  -H "Content-Type: application/json" \\\n  -d '{"example": "data"}'
```

---

#### POST `/api/v1/strategies/{strategy_id}/stop`

**Résumé**: Stop Strategy

**Description**: Arrête une stratégie.

**Paramètres**:
- `strategy_id` (string) (requis): ID de la stratégie

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X POST http://localhost:8000/api/v1/strategies/{strategy_id}/stop \\\n  -H "Content-Type: application/json" \\\n  -d '{"example": "data"}'
```

---

#### GET `/api/v1/strategies/{strategy_id}/performance`

**Résumé**: Get Strategy Performance

**Description**: Récupère les performances d'une stratégie.

**Paramètres**:
- `strategy_id` (string) (requis): ID de la stratégie
- `start_date` (string) (optionnel): Date de début
- `end_date` (string) (optionnel): Date de fin

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/strategies/{strategy_id}/performance
```

---

#### GET `/api/v1/strategies/types`

**Résumé**: Get Strategy Types

**Description**: Récupère les types de stratégies supportés.

**Réponses**:
- `200`: Successful Response

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/strategies/types
```

---

#### POST `/api/v1/strategies/{strategy_id}/backtest`

**Résumé**: Create Backtest

**Description**: Lance un backtest pour une stratégie.

**Paramètres**:
- `strategy_id` (string) (requis): ID de la stratégie

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X POST http://localhost:8000/api/v1/strategies/{strategy_id}/backtest \\\n  -H "Content-Type: application/json" \\\n  -d '{"example": "data"}'
```

---

#### GET `/api/v1/strategies/{strategy_id}/backtests`

**Résumé**: Get Strategy Backtests

**Description**: Récupère les backtests d'une stratégie.

**Paramètres**:
- `strategy_id` (string) (requis): ID de la stratégie
- `page` (integer) (optionnel): Numéro de page
- `per_page` (integer) (optionnel): Backtests par page
- `status` (string) (optionnel): Filtrer par statut

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/strategies/{strategy_id}/backtests
```

---

#### GET `/api/v1/strategies/backtests/{backtest_id}`

**Résumé**: Get Backtest

**Description**: Récupère un backtest spécifique.

**Paramètres**:
- `backtest_id` (string) (requis): ID du backtest

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/strategies/backtests/{backtest_id}
```

---

#### DELETE `/api/v1/strategies/backtests/{backtest_id}`

**Résumé**: Cancel Backtest

**Description**: Annule un backtest en cours.

**Paramètres**:
- `backtest_id` (string) (requis): ID du backtest

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X DELETE http://localhost:8000/api/v1/strategies/backtests/{backtest_id}
```

---

#### GET `/api/v1/strategies/backtests/{backtest_id}/results`

**Résumé**: Get Backtest Results

**Description**: Récupère les résultats détaillés d'un backtest.

**Paramètres**:
- `backtest_id` (string) (requis): ID du backtest
- `include_trades` (boolean) (optionnel): Inclure les détails des trades

**Réponses**:
- `200`: Successful Response
- `422`: Validation Error

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/strategies/backtests/{backtest_id}/results
```

---

### Autres

#### GET `/`

**Résumé**: Root

**Description**: Point d'entrée principal de l'API.

**Réponses**:
- `200`: Successful Response

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/
```

---

#### GET `/health`

**Résumé**: Health Check

**Description**: Endpoint de vérification de santé.

**Réponses**:
- `200`: Successful Response

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/health
```

---

#### GET `/api/v1/status`

**Résumé**: Api Status

**Description**: Statut détaillé de l'API.

**Réponses**:
- `200`: Successful Response

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/status
```

---

#### GET `/api/v1/config`

**Résumé**: Get Config

**Description**: Configuration publique de l'API.

**Réponses**:
- `200`: Successful Response

**Exemple cURL**:
```bash
curl -X GET http://localhost:8000/api/v1/config
```

---

## Modèles de Données

### ApiResponse

Réponse API générique.

**Propriétés**:
- `success` (boolean): 
- `message` (unknown): 
- `data` (unknown): 
- `timestamp` (string): 
- `errors` (unknown): 

### BacktestRequest

Requête de backtest.

**Propriétés**:
- `strategy_id` (string): ID de la stratégie
- `start_date` (string): Date de début du backtest
- `end_date` (string): Date de fin du backtest
- `initial_capital` (number): Capital initial
- `symbols` (unknown): Symboles spécifiques
- `parameters` (unknown): Paramètres personnalisés

### BulkOrderRequest

Requête d'ordres en lot.

**Propriétés**:
- `orders` (array): Liste d'ordres
- `fail_on_error` (boolean): Arrêter si une erreur survient

### CreateOrderRequest

Requête de création d'ordre.

**Propriétés**:
- `symbol` (string): Symbole de trading (ex: BTC/USDT)
- `side` (unknown): Côté de l'ordre
- `type` (unknown): Type d'ordre
- `quantity` (number): Quantité à trader
- `price` (unknown): Prix limite (requis pour LIMIT)
- `stop_price` (unknown): Prix stop
- `time_in_force` (unknown): Durée de validité
- `stop_loss` (unknown): Prix de stop loss
- `take_profit` (unknown): Prix de take profit
- `client_order_id` (unknown): ID client personnalisé
- `notes` (unknown): Notes sur l'ordre

### CreateStrategyRequest

Requête de création de stratégie.

**Propriétés**:
- `name` (string): Nom de la stratégie
- `type` (string): Type de stratégie
- `parameters` (object): Paramètres de la stratégie
- `symbols` (array): Symboles à trader
- `risk_parameters` (unknown): Paramètres de risque
- `active` (boolean): Stratégie active

### HTTPValidationError

**Propriétés**:
- `detail` (array): 

### HealthResponse

Réponse de vérification de santé.

**Propriétés**:
- `status` (unknown): 
- `timestamp` (string): 
- `services` (object): 
- `version` (string): 
- `uptime` (string): 
- `error` (unknown): 

### MarketDataRequest

Requête de données de marché.

**Propriétés**:
- `symbols` (array): Symboles demandés
- `timeframe` (unknown): Timeframe pour historique
- `start_date` (unknown): Date de début
- `end_date` (unknown): Date de fin
- `limit` (unknown): Nombre maximum de points

### OrderSideEnum

Côtés d'ordre.

### OrderTypeEnum

Types d'ordre.

### PaginatedResponse

Réponse paginée.

**Propriétés**:
- `data` (array): 
- `total` (integer): 
- `page` (integer): 
- `per_page` (integer): 
- `pages` (integer): 
- `has_next` (boolean): 
- `has_prev` (boolean): 

### RiskConfigRequest

Requête de configuration des risques.

**Propriétés**:
- `max_portfolio_var` (unknown): VaR maximum du portefeuille
- `max_position_size` (unknown): Taille maximum d'une position
- `max_leverage` (unknown): Leverage maximum
- `max_correlation` (unknown): Corrélation maximum
- `max_drawdown` (unknown): Drawdown maximum
- `position_limit_pct` (unknown): % maximum du portefeuille par position
- `daily_loss_limit` (unknown): Perte journalière maximum

### StatusEnum

Statuts de santé de l'API.

### TimeframeEnum

Timeframes pour les données de marché.

### UpdateOrderRequest

Requête de modification d'ordre.

**Propriétés**:
- `quantity` (unknown): Nouvelle quantité
- `price` (unknown): Nouveau prix
- `stop_price` (unknown): Nouveau prix stop
- `stop_loss` (unknown): Nouveau stop loss
- `take_profit` (unknown): Nouveau take profit

### ValidationError

**Propriétés**:
- `loc` (array): 
- `msg` (string): 
- `type` (string): 

