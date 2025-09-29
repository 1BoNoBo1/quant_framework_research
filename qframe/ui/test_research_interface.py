#!/usr/bin/env python3
"""
Test script pour valider l'interface de recherche QFrame Phase 1
"""
import sys
import os
from pathlib import Path

# Ajouter le chemin vers streamlit_app
current_dir = Path(__file__).parent
streamlit_app_dir = current_dir / "streamlit_app"
sys.path.append(str(streamlit_app_dir))

def test_imports():
    """Test des imports des composants de recherche."""
    print("🧪 Testing Research Interface Imports...")

    try:
        # Test des pages principales
        print("  ✓ Testing main pages...")
        import pages
        print("    ✓ Pages package imported")

        # Test des composants de recherche
        print("  ✓ Testing research components...")
        from components.research.rl_alpha_trainer import RLAlphaTrainer
        from components.research.dmn_lstm_trainer import DMNLSTMTrainer
        from components.research.alpha_formula_visualizer import AlphaFormulaVisualizer
        from components.research.symbolic_operator_builder import SymbolicOperatorBuilder
        print("    ✓ All research components imported successfully")

        # Test des utilitaires ML
        print("  ✓ Testing ML utilities...")
        from utils.ml_utils import MLUtils, ResearchStateManager
        print("    ✓ ML utilities imported successfully")

        # Test des API client existants
        print("  ✓ Testing existing components...")
        from api_client import APIClient
        from components.utils import init_session_state
        from components.charts import create_line_chart
        from components.tables import create_data_table
        print("    ✓ Existing components imported successfully")

        print("✅ All imports successful!")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_component_instantiation():
    """Test de l'instanciation des composants."""
    print("\n🧪 Testing Component Instantiation...")

    try:
        from components.research.rl_alpha_trainer import RLAlphaTrainer
        from components.research.dmn_lstm_trainer import DMNLSTMTrainer
        from components.research.alpha_formula_visualizer import AlphaFormulaVisualizer
        from components.research.symbolic_operator_builder import SymbolicOperatorBuilder
        from utils.ml_utils import MLUtils, ResearchStateManager

        # Test d'instanciation
        print("  ✓ Instantiating RL Alpha Trainer...")
        rl_trainer = RLAlphaTrainer()
        print("    ✓ RLAlphaTrainer created successfully")

        print("  ✓ Instantiating DMN LSTM Trainer...")
        lstm_trainer = DMNLSTMTrainer()
        print("    ✓ DMNLSTMTrainer created successfully")

        print("  ✓ Instantiating Alpha Formula Visualizer...")
        alpha_viz = AlphaFormulaVisualizer()
        print("    ✓ AlphaFormulaVisualizer created successfully")

        print("  ✓ Instantiating Symbolic Operator Builder...")
        op_builder = SymbolicOperatorBuilder()
        print("    ✓ SymbolicOperatorBuilder created successfully")

        print("  ✓ Testing ML Utils...")
        ml_utils = MLUtils()
        print("    ✓ MLUtils available")

        print("✅ All components instantiated successfully!")
        return True

    except Exception as e:
        print(f"❌ Instantiation error: {e}")
        return False

def test_ml_utils_functionality():
    """Test des fonctionnalités des utilitaires ML."""
    print("\n🧪 Testing ML Utils Functionality...")

    try:
        from utils.ml_utils import MLUtils
        import numpy as np

        print("  ✓ Testing sample data generation...")
        sample_data = MLUtils.generate_sample_ohlcv(periods=100)
        assert len(sample_data) == 100
        assert 'open' in sample_data.columns
        assert 'close' in sample_data.columns
        print("    ✓ Sample OHLCV data generated successfully")

        print("  ✓ Testing alpha metrics calculation...")
        returns = np.random.randn(100) * 0.02
        metrics = MLUtils.calculate_alpha_metrics(returns)
        assert 'sharpe_ratio' in metrics
        assert 'ic' in metrics
        print("    ✓ Alpha metrics calculated successfully")

        print("  ✓ Testing performance chart creation...")
        chart = MLUtils.create_performance_chart(returns)
        assert chart is not None
        print("    ✓ Performance chart created successfully")

        print("  ✓ Testing RL training simulation...")
        rl_data = MLUtils.simulate_rl_training_data(episodes=100)
        assert 'rewards' in rl_data
        assert len(rl_data['rewards']) == 100
        print("    ✓ RL training data simulated successfully")

        print("  ✓ Testing LSTM training simulation...")
        lstm_data = MLUtils.simulate_lstm_training_data(epochs=50)
        assert 'train_losses' in lstm_data
        assert len(lstm_data['train_losses']) == 50
        print("    ✓ LSTM training data simulated successfully")

        print("✅ All ML Utils functionality working!")
        return True

    except Exception as e:
        print(f"❌ ML Utils functionality error: {e}")
        return False

def test_operator_library():
    """Test de la bibliothèque d'opérateurs symboliques."""
    print("\n🧪 Testing Symbolic Operator Library...")

    try:
        from components.research.symbolic_operator_builder import SymbolicOperatorBuilder

        builder = SymbolicOperatorBuilder()

        print("  ✓ Testing operator categories...")
        operators = builder.operators
        expected_categories = ["temporal", "statistical", "cross_sectional", "mathematical", "correlation"]

        for category in expected_categories:
            assert category in operators
            print(f"    ✓ {category} operators available")

        print("  ✓ Testing specific operators...")
        # Test des opérateurs critiques
        assert "delta" in operators["temporal"]
        assert "corr" in operators["correlation"]
        assert "cs_rank" in operators["cross_sectional"]
        print("    ✓ Critical operators present")

        print("  ✓ Testing formula validation...")
        test_formula = "corr(open, volume, 10)"
        validation = builder.validate_formula(test_formula)
        assert validation["valid"] == True
        print("    ✓ Formula validation working")

        print("  ✓ Testing auto-build functionality...")
        components = [
            {'type': 'operator', 'name': 'corr', 'info': operators["correlation"]["corr"]},
            {'type': 'feature', 'value': 'open'},
            {'type': 'feature', 'value': 'volume'},
            {'type': 'time_delta', 'value': '10'}
        ]
        auto_formula = builder.auto_build_formula(components)
        assert "corr" in auto_formula
        print("    ✓ Auto-build functionality working")

        print("✅ Symbolic operator library fully functional!")
        return True

    except Exception as e:
        print(f"❌ Operator library error: {e}")
        return False

def test_alpha_library():
    """Test de la bibliothèque d'alphas."""
    print("\n🧪 Testing Alpha Formula Library...")

    try:
        from components.research.alpha_formula_visualizer import AlphaFormulaVisualizer

        visualizer = AlphaFormulaVisualizer()

        print("  ✓ Testing alpha library structure...")
        library = visualizer.alpha_library
        assert "classic" in library
        assert "rl_generated" in library
        assert "custom" in library
        print("    ✓ Alpha library structure correct")

        print("  ✓ Testing classic alphas...")
        classic_alphas = library["classic"]
        assert len(classic_alphas) > 0
        assert "Alpha006" in [alpha["id"] for alpha in classic_alphas]
        print("    ✓ Classic alphas (Alpha101) available")

        print("  ✓ Testing RL generated alphas...")
        rl_alphas = library["rl_generated"]
        assert len(rl_alphas) > 0
        print("    ✓ RL generated alphas available")

        print("  ✓ Testing alpha analysis...")
        test_alpha = classic_alphas[0]
        structure = visualizer._analyze_formula_structure(test_alpha["formula"])
        assert "operators_used" in structure
        print("    ✓ Alpha analysis functionality working")

        print("✅ Alpha formula library fully functional!")
        return True

    except Exception as e:
        print(f"❌ Alpha library error: {e}")
        return False

def test_file_structure():
    """Test de la structure des fichiers."""
    print("\n🧪 Testing File Structure...")

    try:
        # Vérifier la structure des dossiers
        base_dir = current_dir / "streamlit_app"

        print("  ✓ Checking main directories...")
        assert (base_dir / "pages").exists()
        assert (base_dir / "components").exists()
        assert (base_dir / "utils").exists()
        print("    ✓ Main directories exist")

        print("  ✓ Checking research components directory...")
        research_dir = base_dir / "components" / "research"
        assert research_dir.exists()
        print("    ✓ Research components directory exists")

        print("  ✓ Checking page files...")
        pages_dir = base_dir / "pages"
        research_lab = pages_dir / "06_🧠_Research_Lab.py"
        feature_eng = pages_dir / "07_⚙️_Feature_Engineering.py"
        assert research_lab.exists()
        assert feature_eng.exists()
        print("    ✓ Research pages exist")

        print("  ✓ Checking component files...")
        component_files = [
            "rl_alpha_trainer.py",
            "dmn_lstm_trainer.py",
            "alpha_formula_visualizer.py",
            "symbolic_operator_builder.py"
        ]

        for file_name in component_files:
            file_path = research_dir / file_name
            assert file_path.exists()
            print(f"    ✓ {file_name} exists")

        print("  ✓ Checking utility files...")
        utils_dir = base_dir / "utils"
        ml_utils_file = utils_dir / "ml_utils.py"
        assert ml_utils_file.exists()
        print("    ✓ ML utils file exists")

        print("✅ File structure is complete!")
        return True

    except Exception as e:
        print(f"❌ File structure error: {e}")
        return False

def run_integration_test():
    """Test d'intégration complet."""
    print("\n🧪 Running Integration Test...")

    try:
        # Simuler une session utilisateur complète
        from components.research.rl_alpha_trainer import RLAlphaTrainer
        from components.research.symbolic_operator_builder import SymbolicOperatorBuilder
        from utils.ml_utils import MLUtils, ResearchStateManager

        print("  ✓ Simulating user workflow...")

        # 1. Initialiser la session de recherche
        ResearchStateManager.init_research_session()
        print("    ✓ Research session initialized")

        # 2. Créer un trainer RL
        rl_trainer = RLAlphaTrainer()
        print("    ✓ RL trainer created")

        # 3. Créer un constructeur d'opérateurs
        op_builder = SymbolicOperatorBuilder()
        print("    ✓ Operator builder created")

        # 4. Générer des données de test
        sample_data = MLUtils.generate_sample_ohlcv(100)
        returns = sample_data['close'].pct_change().dropna().values
        print("    ✓ Sample data generated")

        # 5. Calculer des métriques
        metrics = MLUtils.calculate_alpha_metrics(returns)
        print("    ✓ Metrics calculated")

        # 6. Créer des graphiques
        chart = MLUtils.create_performance_chart(returns)
        print("    ✓ Charts created")

        # 7. Valider une formule
        formula = "corr(open, volume, 10)"
        validation = op_builder.validate_formula(formula)
        assert validation["valid"]
        print("    ✓ Formula validation successful")

        print("✅ Integration test passed!")
        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Fonction principale de test."""
    print("🚀 QFrame Research Interface - Phase 1 Test Suite")
    print("=" * 60)

    tests = [
        ("File Structure", test_file_structure),
        ("Component Imports", test_imports),
        ("Component Instantiation", test_component_instantiation),
        ("ML Utils Functionality", test_ml_utils_functionality),
        ("Operator Library", test_operator_library),
        ("Alpha Library", test_alpha_library),
        ("Integration Test", run_integration_test)
    ]

    results = []
    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! Phase 1 Research Interface is ready!")
        print("\n📋 What's Available:")
        print("  • RL Alpha Generation interface")
        print("  • DMN LSTM Training interface")
        print("  • Feature Engineering Studio")
        print("  • Symbolic Operator Builder")
        print("  • Alpha Formula Library")
        print("  • Complete ML utilities")
        print("\n🚀 Ready to launch: cd streamlit_app && streamlit run main.py")
    else:
        print(f"⚠️ {total - passed} test(s) failed. Please check the errors above.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)