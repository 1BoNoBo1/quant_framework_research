-- Migration: Create orders table
-- Description: Table pour stocker les ordres de trading avec toutes leurs métadonnées

CREATE TABLE IF NOT EXISTS orders (
    -- Identifiants
    id VARCHAR(36) PRIMARY KEY,
    strategy_id VARCHAR(36),
    parent_order_id VARCHAR(36),

    -- Informations de base
    symbol VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    order_type VARCHAR(20) NOT NULL CHECK (order_type IN ('MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT', 'ICEBERG', 'TWAP', 'VWAP')),

    -- Quantités et prix
    quantity DECIMAL(20, 8) NOT NULL,
    price DECIMAL(20, 8),
    stop_price DECIMAL(20, 8),
    filled_quantity DECIMAL(20, 8) DEFAULT 0,
    remaining_quantity DECIMAL(20, 8) DEFAULT 0,
    average_fill_price DECIMAL(20, 8),

    -- Métadonnées d'ordre
    time_in_force VARCHAR(10) CHECK (time_in_force IN ('DAY', 'GTC', 'IOC', 'FOK')),
    status VARCHAR(20) NOT NULL CHECK (status IN ('PENDING', 'SUBMITTED', 'PARTIAL_FILLED', 'FILLED', 'CANCELLED', 'REJECTED')),

    -- Informations financières
    commission DECIMAL(20, 8),
    fees DECIMAL(20, 8),

    -- Métadonnées flexibles
    tags JSONB,
    executions JSONB DEFAULT '[]'::jsonb,

    -- Horodatages
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    submitted_at TIMESTAMP WITH TIME ZONE,
    filled_at TIMESTAMP WITH TIME ZONE,
    cancelled_at TIMESTAMP WITH TIME ZONE,
    rejected_at TIMESTAMP WITH TIME ZONE,

    -- Raison de rejet
    rejection_reason TEXT,

    -- Contraintes
    CONSTRAINT orders_quantity_positive CHECK (quantity > 0),
    CONSTRAINT orders_price_positive CHECK (price IS NULL OR price > 0),
    CONSTRAINT orders_stop_price_positive CHECK (stop_price IS NULL OR stop_price > 0),
    CONSTRAINT orders_filled_quantity_valid CHECK (filled_quantity >= 0 AND filled_quantity <= quantity),
    CONSTRAINT orders_remaining_quantity_valid CHECK (remaining_quantity >= 0 AND remaining_quantity <= quantity),
    CONSTRAINT orders_quantities_sum CHECK (filled_quantity + remaining_quantity = quantity)
);

-- Index pour les requêtes courantes
CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_strategy_id ON orders(strategy_id);
CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at);
CREATE INDEX IF NOT EXISTS idx_orders_symbol_status ON orders(symbol, status);
CREATE INDEX IF NOT EXISTS idx_orders_strategy_status ON orders(strategy_id, status);

-- Index pour les ordres actifs (les plus consultés)
CREATE INDEX IF NOT EXISTS idx_orders_active ON orders(status) WHERE status IN ('PENDING', 'PARTIAL_FILLED');

-- Index GIN pour les tags JSONB
CREATE INDEX IF NOT EXISTS idx_orders_tags ON orders USING GIN(tags);

-- Index pour les recherches par plage de dates
CREATE INDEX IF NOT EXISTS idx_orders_created_at_desc ON orders(created_at DESC);

-- Trigger pour mettre à jour automatiquement updated_at
CREATE OR REPLACE FUNCTION update_orders_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER trigger_orders_updated_at
    BEFORE UPDATE ON orders
    FOR EACH ROW
    EXECUTE FUNCTION update_orders_updated_at();

-- Contraintes de référence (si les tables existent)
-- ALTER TABLE orders ADD CONSTRAINT fk_orders_strategy
--     FOREIGN KEY (strategy_id) REFERENCES strategies(id) ON DELETE SET NULL;

-- Commentaires pour la documentation
COMMENT ON TABLE orders IS 'Table principale pour stocker tous les ordres de trading';
COMMENT ON COLUMN orders.id IS 'Identifiant unique de l''ordre (UUID)';
COMMENT ON COLUMN orders.symbol IS 'Symbole tradé (ex: AAPL, BTC-USD)';
COMMENT ON COLUMN orders.side IS 'Direction de l''ordre: BUY ou SELL';
COMMENT ON COLUMN orders.order_type IS 'Type d''ordre: MARKET, LIMIT, STOP, etc.';
COMMENT ON COLUMN orders.quantity IS 'Quantité totale de l''ordre';
COMMENT ON COLUMN orders.filled_quantity IS 'Quantité déjà exécutée';
COMMENT ON COLUMN orders.remaining_quantity IS 'Quantité restant à exécuter';
COMMENT ON COLUMN orders.executions IS 'Liste des exécutions partielles (JSON)';
COMMENT ON COLUMN orders.tags IS 'Métadonnées flexibles pour l''ordre (JSON)';
COMMENT ON COLUMN orders.status IS 'Statut actuel de l''ordre';