-- Initialisation de la base de données QFrame
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Tables de base (à adapter selon votre schéma)
CREATE TABLE IF NOT EXISTS portfolios (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    initial_capital DECIMAL(20, 8) NOT NULL,
    base_currency VARCHAR(10) DEFAULT 'USD',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS strategies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100) NOT NULL,
    portfolio_id UUID REFERENCES portfolios(id),
    status VARCHAR(50) DEFAULT 'inactive',
    config JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    portfolio_id UUID REFERENCES portfolios(id),
    strategy_id UUID REFERENCES strategies(id),
    symbol VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL,
    order_type VARCHAR(20) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    price DECIMAL(20, 8),
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes pour performance
CREATE INDEX IF NOT EXISTS idx_portfolios_created_at ON portfolios(created_at);
CREATE INDEX IF NOT EXISTS idx_strategies_portfolio_id ON strategies(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_orders_portfolio_id ON orders(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at);
