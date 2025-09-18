#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilitaire pour calcul des coûts de transaction réalistes
Basé sur l'expérience réelle de trading crypto
"""

import numpy as np
import pandas as pd
from typing import Dict, Union

def get_realistic_trading_costs(symbol: str, trade_size_usd: float) -> Dict[str, float]:
    """
    Calcule les coûts réalistes de trading selon l'asset et la taille
    
    Args:
        symbol: Symbol de trading (ex: BTCUSDT, ETHUSDT)
        trade_size_usd: Taille du trade en USD
        
    Returns:
        Dict avec tous les coûts détaillés
    """
    
    # Coûts de base Binance (avec BNB discount)
    base_costs = {
        "binance_spot_maker": 0.001,     # 0.1%
        "binance_spot_taker": 0.001,     # 0.1%
        "binance_futures_maker": 0.0002, # 0.02%
        "binance_futures_taker": 0.0004, # 0.04%
    }
    
    # Impact de marché selon l'asset
    market_impact = {
        "BTCUSDT": 0.0015,   # 0.15%
        "ETHUSDT": 0.002,    # 0.2%
        "ADAUSDT": 0.004,    # 0.4%
        "SOLUSDT": 0.004,    # 0.4%
        "default": 0.005     # 0.5% pour autres altcoins
    }
    
    # Slippage selon la taille du trade
    def calculate_slippage(size_usd: float) -> float:
        if size_usd < 1000:
            return 0.001       # 0.1% pour petits trades
        elif size_usd < 10000:
            return 0.002       # 0.2% pour trades moyens
        elif size_usd < 50000:
            return 0.004       # 0.4% pour gros trades
        else:
            return 0.008       # 0.8% pour très gros trades
    
    # Funding costs (futures)
    funding_costs = {
        "avg_funding_rate": 0.0001,  # 0.01% par 8h
        "funding_frequency": 3,      # 3 fois par jour
        "daily_funding": 0.0003      # 0.03% par jour en moyenne
    }
    
    # Calculs pour le symbol spécifique
    symbol_impact = market_impact.get(symbol, market_impact["default"])
    trade_slippage = calculate_slippage(trade_size_usd)
    
    # Coût total pour un round-trip (entrée + sortie)
    spot_round_trip = (base_costs["binance_spot_taker"] * 2 + 
                      symbol_impact + trade_slippage)
    
    futures_round_trip = (base_costs["binance_futures_taker"] * 2 + 
                         symbol_impact + trade_slippage + 
                         funding_costs["daily_funding"])
    
    return {
        "spot_entry": base_costs["binance_spot_taker"],
        "spot_exit": base_costs["binance_spot_taker"], 
        "futures_entry": base_costs["binance_futures_taker"],
        "futures_exit": base_costs["binance_futures_taker"],
        "market_impact": symbol_impact,
        "slippage": trade_slippage,
        "funding_daily": funding_costs["daily_funding"],
        "spot_round_trip": spot_round_trip,
        "futures_round_trip": futures_round_trip,
        "total_round_trip": max(spot_round_trip, futures_round_trip)
    }

def apply_realistic_costs_to_backtest(returns: pd.Series, 
                                    signals: pd.Series,
                                    symbol: str = "BTCUSDT",
                                    capital: float = 50000) -> pd.Series:
    """
    Applique les coûts réalistes à une série de returns de backtest
    
    Args:
        returns: Série des returns bruts de la stratégie
        signals: Série des signaux de trading (1, 0, -1)
        symbol: Symbol tradé
        capital: Capital de base pour calculer taille trades
        
    Returns:
        Série des returns nets après coûts réalistes
    """
    
    if len(returns) != len(signals):
        raise ValueError("Returns et signals doivent avoir la même longueur")
    
    # Détection des trades (changement de signal)
    signal_changes = signals.diff().abs()
    trade_occurred = signal_changes > 0
    
    # Calcul de la taille moyenne des trades
    avg_position_size = abs(signals).mean() * capital
    
    # Coûts réalistes selon le symbol
    costs = get_realistic_trading_costs(symbol, avg_position_size)
    total_cost_per_trade = costs["total_round_trip"]
    
    # Application des coûts
    cost_series = pd.Series(0.0, index=returns.index)
    cost_series[trade_occurred] = total_cost_per_trade
    
    # Returns nets
    net_returns = returns - cost_series
    
    return net_returns

def calculate_minimum_profit_target(symbol: str, capital: float = 50000) -> Dict[str, float]:
    """
    Calcule les profits minimum requis pour être rentable
    
    Args:
        symbol: Symbol de trading
        capital: Capital de trading
        
    Returns:
        Dict avec les targets minimum
    """
    
    # Position size typique (1% du capital)
    typical_position = capital * 0.01
    
    costs = get_realistic_trading_costs(symbol, typical_position)
    
    # Il faut au minimum couvrir les coûts + marge de sécurité
    min_profit_per_trade = costs["total_round_trip"] * 2  # 2x les coûts
    min_daily_profit = min_profit_per_trade * 2           # 2 trades par jour
    min_monthly_profit = min_daily_profit * 22            # 22 jours de trading
    
    return {
        "min_profit_per_trade_pct": min_profit_per_trade,
        "min_daily_profit_pct": min_daily_profit,
        "min_monthly_profit_pct": min_monthly_profit,
        "min_profit_per_trade_usd": min_profit_per_trade * capital,
        "min_daily_profit_usd": min_daily_profit * capital,
        "min_monthly_profit_usd": min_monthly_profit * capital,
        "total_cost_per_trade": costs["total_round_trip"],
        "breakeven_win_rate": 0.5 + (costs["total_round_trip"] / 0.02)  # Assuming 2% avg win
    }

def validate_strategy_viability(sharpe: float, 
                              hit_rate: float,
                              avg_return_per_trade: float,
                              symbol: str = "BTCUSDT") -> Dict[str, Union[bool, str]]:
    """
    Valide si une stratégie est viable avec les coûts réels
    
    Args:
        sharpe: Ratio de Sharpe de la stratégie
        hit_rate: Taux de réussite (0-1)
        avg_return_per_trade: Return moyen par trade
        symbol: Symbol tradé
        
    Returns:
        Dict avec verdict de viabilité
    """
    
    targets = calculate_minimum_profit_target(symbol)
    
    viability_checks = {
        "sharpe_viable": sharpe > 1.0,
        "hit_rate_viable": hit_rate > targets["breakeven_win_rate"],
        "profit_per_trade_viable": avg_return_per_trade > targets["min_profit_per_trade_pct"],
        "overall_viable": False
    }
    
    # Verdict global
    viability_checks["overall_viable"] = all([
        viability_checks["sharpe_viable"],
        viability_checks["hit_rate_viable"],
        viability_checks["profit_per_trade_viable"]
    ])
    
    # Messages d'explication
    messages = []
    if not viability_checks["sharpe_viable"]:
        messages.append(f"❌ Sharpe {sharpe:.2f} < 1.0 requis")
    if not viability_checks["hit_rate_viable"]:
        messages.append(f"❌ Hit rate {hit_rate:.1%} < {targets['breakeven_win_rate']:.1%} requis")
    if not viability_checks["profit_per_trade_viable"]:
        messages.append(f"❌ Profit/trade {avg_return_per_trade:.3f} < {targets['min_profit_per_trade_pct']:.3f} requis")
    
    if viability_checks["overall_viable"]:
        messages.append("✅ Stratégie viable avec coûts réalistes")
    
    viability_checks["messages"] = messages
    viability_checks["required_targets"] = targets
    
    return viability_checks

# Exemples d'utilisation
if __name__ == "__main__":
    
    # Test 1: Coûts pour BTCUSDT
    costs_btc = get_realistic_trading_costs("BTCUSDT", 5000)
    print("=== COÛTS BTC (5k$ trade) ===")
    for key, value in costs_btc.items():
        print(f"{key}: {value:.4f} ({value*100:.2f}%)")
    
    print("\n=== TARGETS MINIMUM ===")
    targets = calculate_minimum_profit_target("BTCUSDT", 50000)
    for key, value in targets.items():
        if "pct" in key:
            print(f"{key}: {value:.4f} ({value*100:.2f}%)")
        elif "usd" in key:
            print(f"{key}: ${value:.2f}")
        else:
            print(f"{key}: {value:.4f}")
    
    print("\n=== VALIDATION STRATÉGIE ===")
    validation = validate_strategy_viability(0.5, 0.48, 0.001, "BTCUSDT")
    for msg in validation["messages"]:
        print(msg)