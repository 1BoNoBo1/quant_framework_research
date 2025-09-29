#!/usr/bin/env python3
"""
📚 Générateur de Documentation API
Génère la documentation OpenAPI complète et des exemples d'utilisation
"""

import json
import sys
import subprocess
import time
import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_openapi_schema():
    """Génère le schéma OpenAPI en démarrant l'API temporairement."""

    # Démarrer l'API en arrière-plan
    logger.info("🚀 Démarrage temporaire de l'API pour génération du schéma...")
    api_process = subprocess.Popen([
        "poetry", "run", "python", "start_api.py", "--port", "8004"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        # Attendre le démarrage
        time.sleep(8)

        # Récupérer le schéma OpenAPI
        response = requests.get("http://localhost:8004/openapi.json", timeout=10)

        if response.status_code == 200:
            schema = response.json()

            # Sauvegarder le schéma OpenAPI
            schema_file = Path("docs/api_schema.json")
            schema_file.parent.mkdir(exist_ok=True)

            with open(schema_file, "w") as f:
                json.dump(schema, f, indent=2)

            logger.info(f"✅ Schéma OpenAPI sauvegardé: {schema_file}")

            # Générer la documentation Markdown
            generate_markdown_docs(schema)

            return schema

        else:
            logger.error(f"❌ Erreur récupération schéma: HTTP {response.status_code}")
            return None

    except Exception as e:
        logger.error(f"❌ Erreur génération schéma: {e}")
        return None

    finally:
        # Arrêter l'API
        logger.info("🛑 Arrêt de l'API temporaire...")
        api_process.terminate()
        api_process.wait(timeout=5)

def generate_markdown_docs(schema):
    """Génère une documentation Markdown à partir du schéma OpenAPI."""

    markdown_content = f"""# 📚 Documentation API QFrame

## Informations Générales

- **Titre**: {schema.get('info', {}).get('title', 'QFrame API')}
- **Version**: {schema.get('info', {}).get('version', '1.0.0')}
- **Description**: {schema.get('info', {}).get('description', 'Backend API pour QFrame')}
- **URL de base**: `http://localhost:8000`

## Authentification

L'API utilise un middleware d'authentification basique. Pour les tests de développement, aucune authentification n'est requise sur les endpoints publics.

## Endpoints Disponibles

"""

    # Grouper les endpoints par tags
    paths = schema.get('paths', {})
    endpoints_by_tag = {}

    for path, methods in paths.items():
        for method, details in methods.items():
            if method.lower() in ['get', 'post', 'put', 'delete', 'patch']:
                tags = details.get('tags', ['Autres'])
                tag = tags[0] if tags else 'Autres'

                if tag not in endpoints_by_tag:
                    endpoints_by_tag[tag] = []

                endpoints_by_tag[tag].append({
                    'path': path,
                    'method': method.upper(),
                    'summary': details.get('summary', ''),
                    'description': details.get('description', ''),
                    'parameters': details.get('parameters', []),
                    'responses': details.get('responses', {})
                })

    # Générer la documentation par section
    for tag, endpoints in endpoints_by_tag.items():
        markdown_content += f"### {tag}\n\n"

        for endpoint in endpoints:
            markdown_content += f"#### {endpoint['method']} `{endpoint['path']}`\n\n"

            if endpoint['summary']:
                markdown_content += f"**Résumé**: {endpoint['summary']}\n\n"

            if endpoint['description']:
                markdown_content += f"**Description**: {endpoint['description']}\n\n"

            # Paramètres
            if endpoint['parameters']:
                markdown_content += "**Paramètres**:\n"
                for param in endpoint['parameters']:
                    param_name = param.get('name', '')
                    param_type = param.get('schema', {}).get('type', 'string')
                    param_desc = param.get('description', '')
                    param_required = param.get('required', False)
                    required_text = " (requis)" if param_required else " (optionnel)"

                    markdown_content += f"- `{param_name}` ({param_type}){required_text}: {param_desc}\n"
                markdown_content += "\n"

            # Réponses
            if endpoint['responses']:
                markdown_content += "**Réponses**:\n"
                for code, response in endpoint['responses'].items():
                    desc = response.get('description', '')
                    markdown_content += f"- `{code}`: {desc}\n"
                markdown_content += "\n"

            # Exemple cURL
            curl_example = generate_curl_example(endpoint)
            markdown_content += f"**Exemple cURL**:\n```bash\n{curl_example}\n```\n\n"

            markdown_content += "---\n\n"

    # Modèles de données
    if 'components' in schema and 'schemas' in schema['components']:
        markdown_content += "## Modèles de Données\n\n"

        for model_name, model_schema in schema['components']['schemas'].items():
            markdown_content += f"### {model_name}\n\n"

            if 'description' in model_schema:
                markdown_content += f"{model_schema['description']}\n\n"

            if 'properties' in model_schema:
                markdown_content += "**Propriétés**:\n"
                for prop_name, prop_details in model_schema['properties'].items():
                    prop_type = prop_details.get('type', 'unknown')
                    prop_desc = prop_details.get('description', '')
                    markdown_content += f"- `{prop_name}` ({prop_type}): {prop_desc}\n"
                markdown_content += "\n"

    # Sauvegarder la documentation
    docs_file = Path("docs/API_DOCUMENTATION.md")
    with open(docs_file, "w") as f:
        f.write(markdown_content)

    logger.info(f"✅ Documentation Markdown générée: {docs_file}")

def generate_curl_example(endpoint):
    """Génère un exemple cURL pour un endpoint."""
    method = endpoint['method']
    path = endpoint['path']

    if method == 'GET':
        return f"curl -X GET http://localhost:8000{path}"
    elif method == 'POST':
        return f'curl -X POST http://localhost:8000{path} \\\\\\n  -H "Content-Type: application/json" \\\\\\n  -d \'{{"example": "data"}}\''
    elif method == 'PUT':
        return f'curl -X PUT http://localhost:8000{path} \\\\\\n  -H "Content-Type: application/json" \\\\\\n  -d \'{{"example": "data"}}\''
    elif method == 'DELETE':
        return f"curl -X DELETE http://localhost:8000{path}"
    else:
        return f"curl -X {method} http://localhost:8000{path}"

def generate_usage_examples():
    """Génère des exemples d'utilisation en Python."""

    examples_content = '''# 🐍 Exemples d'Utilisation API QFrame

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
curl -X POST http://localhost:8000/api/v1/orders \\
  -H "Content-Type: application/json" \\
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
'''

    examples_file = Path("docs/API_EXAMPLES.md")
    with open(examples_file, "w") as f:
        f.write(examples_content)

    logger.info(f"✅ Exemples d'utilisation générés: {examples_file}")

def main():
    """Point d'entrée principal."""
    logger.info("📚 Génération de la documentation API QFrame...")

    # Créer le dossier docs
    Path("docs").mkdir(exist_ok=True)

    # Générer le schéma OpenAPI et la documentation
    schema = generate_openapi_schema()

    if schema:
        logger.info("✅ Documentation OpenAPI générée avec succès")

        # Générer les exemples d'utilisation
        generate_usage_examples()

        logger.info("🎉 Documentation complète générée !")
        logger.info("📁 Fichiers créés:")
        logger.info("  - docs/api_schema.json (Schéma OpenAPI)")
        logger.info("  - docs/API_DOCUMENTATION.md (Documentation complète)")
        logger.info("  - docs/API_EXAMPLES.md (Exemples d'utilisation)")

    else:
        logger.error("❌ Échec de la génération de documentation")
        sys.exit(1)

if __name__ == "__main__":
    main()