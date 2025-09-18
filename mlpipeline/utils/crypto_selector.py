#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sélecteur interactif de symboles crypto
Permet de choisir facilement parmi les principales crypto-monnaies
"""

import sys
from typing import List, Dict, Set

class CryptoSelector:
    """Sélecteur interactif de symboles crypto avec catégories"""

    def __init__(self):
        self.symbols = {
            "🔥 Top 10": [
                "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
                "SOLUSDT", "DOTUSDT", "DOGEUSDT", "AVAXUSDT", "MATICUSDT"
            ],
            "💎 Layer 1": [
                "BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "DOTUSDT",
                "AVAXUSDT", "ATOMUSDT", "NEARUSDT", "ALGOUSDT", "EGLDUSDT",
                "FILUSDT", "ICPUSDT", "FLOWUSDT", "HBARUSDT", "VETUSDT"
            ],
            "⚡ Layer 2 & Scaling": [
                "MATICUSDT", "OPUSDT", "ARBUSDT", "IMXUSDT", "LRCUSDT",
                "SKLUSDT", "OMGUSDT", "STRKUSDT"
            ],
            "🏦 DeFi": [
                "UNIUSDT", "AAVEUSDT", "COMPUSDT", "MKRUSDT", "SUSHIUSDT",
                "CRVUSDT", "1INCHUSDT", "YFIUSDT", "SNXUSDT", "BALUSDT",
                "PANCAKEUSDT", "CAKEUSDT", "ALPHAUSDT", "DYDXUSDT"
            ],
            "🎮 Gaming & NFT": [
                "AXSUSDT", "SANDUSDT", "MANAUSDT", "ENJUSDT", "CHZUSDT",
                "GALAUSDT", "FLOWUSDT", "THETAUSDT", "ALICEUSDT", "TLMUSDT"
            ],
            "🌐 Infrastructure": [
                "LINKUSDT", "CHZUSDT", "BATUSDT", "GRTUSDT", "RNDRSUDT",
                "STORJUSDT", "SIACOINUSDT", "IOTAUSDT", "OCEANUSDT"
            ],
            "🏪 Exchange Tokens": [
                "BNBUSDT", "CAKEUSDT", "FTMUSDT", "KCSUSDT", "GTUSDT",
                "OKBUSDT", "BTTUSDT"
            ],
            "💰 Stablecoins": [
                "USDCUSDT", "BUSDUSDT", "DAIUSDT", "TUSDUSDT", "FRAXUSDT"
            ],
            "🔮 Oracles & AI": [
                "LINKUSDT", "BANDUSDT", "APIUSDT", "AIUSDT", "FETUSDT",
                "AGIXUSDT", "OCEANUSDT", "RNDRSUDT"
            ],
            "🚀 Nouveaux & Tendances": [
                "SUIUSDT", "APTUSDT", "OPUSDT", "ARBUSDT", "STRKUSDT",
                "BLURUSDT", "LDOUSDT", "MAGICUSDT", "GMXUSDT", "INJUSDT"
            ]
        }

        self.timeframes = {
            "📊 Timeframes Standard": ["1m", "5m", "15m", "1h", "4h", "1d", "1w"],
            "⚡ High Frequency": ["1m", "3m", "5m", "15m", "30m"],
            "📈 Swing Trading": ["1h", "4h", "12h", "1d"],
            "💼 Position Trading": ["4h", "1d", "3d", "1w", "1M"]
        }

    def display_categories(self) -> Dict[str, List[str]]:
        """Affiche toutes les catégories disponibles"""
        print("\n" + "="*60)
        print("🚀 SÉLECTEUR CRYPTO AVANCÉ")
        print("="*60)

        for i, (category, symbols) in enumerate(self.symbols.items(), 1):
            print(f"\n{i}. {category} ({len(symbols)} symboles)")
            # Afficher les 5 premiers symboles en aperçu
            preview = symbols[:5]
            if len(symbols) > 5:
                preview.append(f"... (+{len(symbols)-5})")
            print(f"   {', '.join(preview)}")

        return self.symbols

    def select_symbols_interactive(self) -> List[str]:
        """Sélection interactive de symboles"""
        selected_symbols = set()

        while True:
            self.display_categories()
            print(f"\n📋 Symboles sélectionnés: {len(selected_symbols)}")
            if selected_symbols:
                print(f"   {', '.join(sorted(selected_symbols)[:10])}{'...' if len(selected_symbols) > 10 else ''}")

            print("\n🎯 Options:")
            print("   1-10: Sélectionner une catégorie")
            print("   a: Ajouter symbole personnalisé")
            print("   r: Retirer un symbole")
            print("   c: Effacer sélection")
            print("   s: Afficher tous les symboles sélectionnés")
            print("   t: Sélectionner timeframes")
            print("   f: Terminer et continuer")
            print("   q: Quitter")

            choice = input("\n👉 Votre choix: ").strip().lower()

            if choice == 'q':
                sys.exit(0)
            elif choice == 'f':
                if selected_symbols:
                    return list(selected_symbols)
                else:
                    print("❌ Aucun symbole sélectionné!")
                    continue
            elif choice == 'c':
                selected_symbols.clear()
                print("✅ Sélection effacée")
            elif choice == 's':
                self._show_selected_symbols(selected_symbols)
            elif choice == 'a':
                symbol = self._add_custom_symbol()
                if symbol:
                    selected_symbols.add(symbol)
            elif choice == 'r':
                self._remove_symbol(selected_symbols)
            elif choice == 't':
                return list(selected_symbols)  # Pour l'instant, on retourne pour la sélection timeframes
            elif choice.isdigit():
                cat_num = int(choice)
                if 1 <= cat_num <= len(self.symbols):
                    category = list(self.symbols.keys())[cat_num - 1]
                    self._select_from_category(category, selected_symbols)
                else:
                    print("❌ Numéro de catégorie invalide!")
            else:
                print("❌ Choix invalide!")

    def _select_from_category(self, category: str, selected_symbols: Set[str]):
        """Sélection dans une catégorie spécifique"""
        symbols = self.symbols[category]

        print(f"\n📂 {category}")
        print("-" * 50)

        for i, symbol in enumerate(symbols, 1):
            status = "✅" if symbol in selected_symbols else "  "
            print(f"{status} {i:2d}. {symbol}")

        print("\n🎯 Options:")
        print("   1-N: Toggle symbole")
        print("   all: Sélectionner tous")
        print("   none: Désélectionner tous")
        print("   back: Retour")

        while True:
            choice = input("\n👉 Choix dans catégorie: ").strip().lower()

            if choice == 'back':
                break
            elif choice == 'all':
                selected_symbols.update(symbols)
                print(f"✅ Tous les symboles de {category} ajoutés")
                break
            elif choice == 'none':
                for symbol in symbols:
                    selected_symbols.discard(symbol)
                print(f"✅ Tous les symboles de {category} retirés")
                break
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(symbols):
                    symbol = symbols[idx]
                    if symbol in selected_symbols:
                        selected_symbols.remove(symbol)
                        print(f"❌ {symbol} retiré")
                    else:
                        selected_symbols.add(symbol)
                        print(f"✅ {symbol} ajouté")
                else:
                    print("❌ Numéro invalide!")
            else:
                print("❌ Choix invalide!")

    def _add_custom_symbol(self) -> str:
        """Ajouter un symbole personnalisé"""
        print("\n📝 Ajouter symbole personnalisé:")
        print("   Format: SYMBOL (ex: LINKUSDT, DOTUSDT)")

        symbol = input("👉 Symbole: ").strip().upper()

        if not symbol:
            return None

        # Validation basique
        if not symbol.endswith('USDT'):
            symbol += 'USDT'

        print(f"✅ Symbole personnalisé: {symbol}")
        return symbol

    def _remove_symbol(self, selected_symbols: Set[str]):
        """Retirer un symbole de la sélection"""
        if not selected_symbols:
            print("❌ Aucun symbole à retirer!")
            return

        print("\n📋 Symboles sélectionnés:")
        symbols_list = sorted(selected_symbols)
        for i, symbol in enumerate(symbols_list, 1):
            print(f"   {i:2d}. {symbol}")

        try:
            choice = input("\n👉 Numéro à retirer (ou 'all' pour tout): ").strip()
            if choice.lower() == 'all':
                selected_symbols.clear()
                print("✅ Tous les symboles retirés")
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(symbols_list):
                    symbol = symbols_list[idx]
                    selected_symbols.remove(symbol)
                    print(f"✅ {symbol} retiré")
                else:
                    print("❌ Numéro invalide!")
        except (ValueError, IndexError):
            print("❌ Entrée invalide!")

    def _show_selected_symbols(self, selected_symbols: Set[str]):
        """Affiche tous les symboles sélectionnés"""
        if not selected_symbols:
            print("❌ Aucun symbole sélectionné!")
            return

        print(f"\n📊 SYMBOLES SÉLECTIONNÉS ({len(selected_symbols)}):")
        print("-" * 50)

        sorted_symbols = sorted(selected_symbols)
        for i, symbol in enumerate(sorted_symbols, 1):
            print(f"   {i:2d}. {symbol}")

        input("\n👉 Appuyez sur Entrée pour continuer...")

    def select_timeframes_interactive(self) -> List[str]:
        """Sélection interactive de timeframes"""
        selected_timeframes = set()

        while True:
            print("\n" + "="*60)
            print("⏰ SÉLECTEUR TIMEFRAMES")
            print("="*60)

            for i, (category, timeframes) in enumerate(self.timeframes.items(), 1):
                print(f"\n{i}. {category}")
                print(f"   {', '.join(timeframes)}")

            print(f"\n📋 Timeframes sélectionnés: {len(selected_timeframes)}")
            if selected_timeframes:
                print(f"   {', '.join(sorted(selected_timeframes))}")

            print("\n🎯 Options:")
            print("   1-4: Sélectionner catégorie timeframes")
            print("   c: Effacer sélection")
            print("   s: Afficher sélection")
            print("   f: Terminer")
            print("   q: Quitter")

            choice = input("\n👉 Votre choix: ").strip().lower()

            if choice == 'q':
                sys.exit(0)
            elif choice == 'f':
                if selected_timeframes:
                    return list(selected_timeframes)
                else:
                    print("❌ Aucun timeframe sélectionné!")
                    continue
            elif choice == 'c':
                selected_timeframes.clear()
                print("✅ Sélection effacée")
            elif choice == 's':
                self._show_selected_timeframes(selected_timeframes)
            elif choice.isdigit():
                cat_num = int(choice)
                if 1 <= cat_num <= len(self.timeframes):
                    category = list(self.timeframes.keys())[cat_num - 1]
                    self._select_timeframes_from_category(category, selected_timeframes)
                else:
                    print("❌ Numéro de catégorie invalide!")
            else:
                print("❌ Choix invalide!")

    def _select_timeframes_from_category(self, category: str, selected_timeframes: Set[str]):
        """Sélection timeframes dans une catégorie"""
        timeframes = self.timeframes[category]

        print(f"\n📂 {category}")
        print("-" * 50)

        for i, tf in enumerate(timeframes, 1):
            status = "✅" if tf in selected_timeframes else "  "
            print(f"{status} {i:2d}. {tf}")

        print("\n🎯 Options:")
        print("   1-N: Toggle timeframe")
        print("   all: Sélectionner tous")
        print("   none: Désélectionner tous")
        print("   back: Retour")

        while True:
            choice = input("\n👉 Choix: ").strip().lower()

            if choice == 'back':
                break
            elif choice == 'all':
                selected_timeframes.update(timeframes)
                print(f"✅ Tous les timeframes de {category} ajoutés")
                break
            elif choice == 'none':
                for tf in timeframes:
                    selected_timeframes.discard(tf)
                print(f"✅ Tous les timeframes de {category} retirés")
                break
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(timeframes):
                    tf = timeframes[idx]
                    if tf in selected_timeframes:
                        selected_timeframes.remove(tf)
                        print(f"❌ {tf} retiré")
                    else:
                        selected_timeframes.add(tf)
                        print(f"✅ {tf} ajouté")
                else:
                    print("❌ Numéro invalide!")
            else:
                print("❌ Choix invalide!")

    def _show_selected_timeframes(self, selected_timeframes: Set[str]):
        """Affiche tous les timeframes sélectionnés"""
        if not selected_timeframes:
            print("❌ Aucun timeframe sélectionné!")
            return

        print(f"\n⏰ TIMEFRAMES SÉLECTIONNÉS ({len(selected_timeframes)}):")
        print("-" * 50)

        sorted_timeframes = sorted(selected_timeframes)
        for i, tf in enumerate(sorted_timeframes, 1):
            print(f"   {i:2d}. {tf}")

        input("\n👉 Appuyez sur Entrée pour continuer...")

def main():
    """Interface principale du sélecteur"""
    selector = CryptoSelector()

    print("🚀 Sélection des symboles crypto...")
    symbols = selector.select_symbols_interactive()

    print("\n⏰ Sélection des timeframes...")
    timeframes = selector.select_timeframes_interactive()

    print("\n" + "="*60)
    print("✅ SÉLECTION TERMINÉE")
    print("="*60)
    print(f"📊 Symboles ({len(symbols)}): {', '.join(symbols)}")
    print(f"⏰ Timeframes ({len(timeframes)}): {', '.join(timeframes)}")

    # Génération commandes Makefile
    print("\n🔧 COMMANDES MAKEFILE:")
    print(f'make clean-symbols SYMBOLS="{" ".join(symbols)}"')
    print(f'make clean-timeframes TIMEFRAMES="{" ".join(timeframes)}"')
    print(f'make clean-selective SYMBOLS="{" ".join(symbols)}" TIMEFRAMES="{" ".join(timeframes)}"')

    return symbols, timeframes

if __name__ == "__main__":
    main()