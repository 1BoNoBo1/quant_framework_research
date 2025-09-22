# Installation de TA-Lib

## Prérequis : Installer la bibliothèque C TA-Lib

TA-Lib Python nécessite la bibliothèque C TA-Lib installée sur votre système.

### Sur Ubuntu/Debian :

```bash
# 1. Installer les dépendances de compilation
sudo apt-get update
sudo apt-get install -y build-essential wget

# 2. Télécharger et compiler TA-Lib
cd /tmp
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib
./configure --prefix=/usr
make
sudo make install

# 3. Mettre à jour les bibliothèques
sudo ldconfig
```

### Alternative plus simple (peut ne pas avoir la dernière version) :

```bash
sudo apt-get install libta-lib-dev
```

## Après l'installation de la bibliothèque C

Une fois la bibliothèque C installée, vous pouvez installer le package Python :

```bash
# Réinstaller avec Poetry
poetry add ta-lib

# Ou si déjà dans pyproject.toml
poetry install
```

## Vérification

```python
# Test que TA-Lib fonctionne
poetry run python -c "import talib; print(talib.__version__)"
```