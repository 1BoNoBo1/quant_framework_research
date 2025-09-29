# 🐍 Exemples d'Utilisation API QFrame

## Client Python Simple

```python
import requests
import json

class QFrameAPIClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def get_health(self):
        """Vérifie l'état de santé de l'API."""
        response = self.session.get(f"{self.base_url}/health")
        return response.json()

    def get_strategies(self):
        """Récupère la liste des stratégies."""
        response = self.session.get(f"{self.base_url}/api/v1/strategies")
        return response.json()

    def create_order(self, symbol, side, order_type, quantity):
        """Crée un nouvel ordre."""
        order_data = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity
        }
        response = self.session.post(
            f"{self.base_url}/api/v1/orders",
            json=order_data
        )
        return response.json()

# Utilisation
client = QFrameAPIClient()

# Vérifier la santé
health = client.get_health()
print("Santé API:", health)

# Récupérer les stratégies
strategies = client.get_strategies()
print("Stratégies:", strategies)

# Créer un ordre
order = client.create_order("BTC/USD", "BUY", "MARKET", 0.001)
print("Ordre créé:", order)
```

## Exemples JavaScript/Node.js

```javascript
const axios = require('axios');

class QFrameAPIClient {
    constructor(baseURL = 'http://localhost:8000') {
        this.client = axios.create({
            baseURL,
            headers: {
                'Content-Type': 'application/json'
            }
        });
    }

    async getHealth() {
        const response = await this.client.get('/health');
        return response.data;
    }

    async getStrategies() {
        const response = await this.client.get('/api/v1/strategies');
        return response.data;
    }

    async createOrder(symbol, side, type, quantity) {
        const orderData = { symbol, side, type, quantity };
        const response = await this.client.post('/api/v1/orders', orderData);
        return response.data;
    }
}

// Utilisation
(async () => {
    const client = new QFrameAPIClient();

    try {
        const health = await client.getHealth();
        console.log('Santé API:', health);

        const strategies = await client.getStrategies();
        console.log('Stratégies:', strategies);

    } catch (error) {
        console.error('Erreur API:', error.response?.data || error.message);
    }
})();
```

## Tests avec cURL

```bash
# Test de santé
curl -X GET http://localhost:8000/health

# Récupérer les stratégies
curl -X GET http://localhost:8000/api/v1/strategies

# Créer un ordre
curl -X POST http://localhost:8000/api/v1/orders \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC/USD",
    "side": "BUY",
    "type": "MARKET",
    "quantity": 0.001
  }'

# Récupérer les positions
curl -X GET http://localhost:8000/api/v1/positions

# Calculer VaR
curl -X GET "http://localhost:8000/api/v1/risk/var?confidence_level=0.95&method=monte_carlo"
```

## WebSocket Temps Réel

```python
import asyncio
import websockets
import json

async def websocket_client():
    uri = "ws://localhost:8000/ws"

    async with websockets.connect(uri) as websocket:
        # S'abonner aux événements de prix
        subscribe_message = {
            "type": "subscribe",
            "event_types": ["PRICE_UPDATE", "ORDER_FILLED"],
            "symbols": ["BTC/USD", "ETH/USD"]
        }

        await websocket.send(json.dumps(subscribe_message))

        # Écouter les messages
        async for message in websocket:
            data = json.loads(message)
            print(f"Reçu: {data}")

# Exécuter le client WebSocket
asyncio.run(websocket_client())
```
