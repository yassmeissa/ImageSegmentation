#!/usr/bin/env python3
"""
Script principal de validation - Lance tous les tests
"""

import sys
import os

# Ajouter le chemin du projet
sys.path.insert(0, '/Users/yassmeissa/Downloads/apptkr_imageprocessing')

from tests.test_models import TestModels
from tests.test_ui import TestUI


def main():
    """Exécuter tous les tests"""
    print("\n" + "="*70)
    print("SUITE COMPLÈTE DE TESTS DE VALIDATION")
    print("="*70)
    
    # Tests des modèles
    print("\n\nPHASE 1: Tests des modèles de clustering")
    model_tester = TestModels()
    model_tester.run_all_tests()
    model_results = (model_tester.passed, model_tester.failed)
    
    # Tests UI
    print("\n\nPHASE 2: Tests de l'interface utilisateur")
    ui_tester = TestUI()
    ui_tester.run_all_tests()
    ui_results = (ui_tester.passed, ui_tester.failed)
    
    # Résumé global
    print("\n" + "="*70)
    print("RÉSUMÉ GLOBAL")
    print("="*70)
    
    total_passed = model_results[0] + ui_results[0]
    total_failed = model_results[1] + ui_results[1]
    total = total_passed + total_failed
    
    print(f"\nTests des modèles: {model_results[0]} réussis, {model_results[1]} échoués")
    print(f"Tests UI: {ui_results[0]} réussis, {ui_results[1]} échoués")
    print(f"\n{'='*70}")
    print(f"[OK] TOTAL RÉUSSIS: {total_passed}/{total}")
    print(f"[ERREUR] TOTAL ÉCHOUÉS: {total_failed}/{total}")
    print(f"{'='*70}")
    
    if total_failed == 0:
        print("\nTOUS LES TESTS SONT PASSÉS!")
        print("\nL'application est prête pour la production!\n")
        return 0
    else:
        print(f"\n{total_failed} test(s) ont échoué - Veuillez corriger les erreurs\n")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
