"""
Master Test Runner for MyoPS-Net MyoPS380 Adaptation E2E Test Suite.
Executes all 5 test modules across 4 tiers covering Features F1 through F11.
Usage:
    python tests/run_all_tests.py
Exit code:
    0 if all tests pass, 1 if any test fails.
"""

import sys
import os
import time
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


TEST_MODULES = [
    "tests.test_data_pipeline",
    "tests.test_target_adapter",
    "tests.test_loss_hardening",
    "tests.test_training_dynamics",
    "tests.test_inference_eval",
]

FEATURE_MAPPING = {
    "F1 (Data Pipeline & Manifests)": ["test_f1_"],
    "F2 (Normalization & Zero-Padding)": ["test_f2_"],
    "F3 (SCAR RandomGenerator)": ["test_f3_"],
    "F4 (Validation Elimination)": ["test_f4_"],
    "F5 (Target Adapter)": ["test_f5_"],
    "F6 (WCELoss Bug Fix & Hardening)": ["test_f6_"],
    "F7 (Optimizer & Scheduler)": ["test_f7_"],
    "F8 (Batch Accumulation & Checkpoint)": ["test_f8_"],
    "F9 (Test Inference Pipeline)": ["test_f9_"],
    "F10 (Shared Evaluator)": ["test_f10_"],
    "F11 (End-to-End Verification)": ["test_f11_"],
}

TIER_MAPPING = {
    "Tier 1 (Feature Coverage)": ["test_f1_", "test_f2_", "test_f3_", "test_f4_", "test_f5_", "test_f6_", "test_f7_", "test_f8_", "test_f9_", "test_f10_", "test_f11_"],
    "Tier 2 (Boundary & Corner Cases)": ["test_f1_", "test_f2_", "test_f3_", "test_f4_", "test_f5_", "test_f6_", "test_f7_", "test_f8_", "test_f9_", "test_f10_", "test_f11_"],
    "Tier 3 (Cross-Feature Combinations)": ["test_t3_"],
    "Tier 4 (Real-World Application Scenarios)": ["test_t4_"],
}


def categorize_test(test_item, test_name: str, test_class_name: str):
    """Categorizes a test into Feature and Tier using docstring and method naming."""
    # Check docstring first
    method = getattr(test_item, test_name, None)
    doc = getattr(method, "__doc__", "") or ""

    tier = "Unknown"
    if "Tier 4:" in doc or test_name.startswith("test_t4_"):
        tier = "Tier 4 (Real-World Application Scenarios)"
    elif "Tier 3:" in doc or test_name.startswith("test_t3_") or "CrossFeature" in test_class_name:
        tier = "Tier 3 (Cross-Feature Combinations)"
    elif "Tier 2:" in doc or "Boundary" in test_class_name or "defect" in test_name.lower():
        tier = "Tier 2 (Boundary & Corner Cases)"
    elif "Tier 1:" in doc or "Coverage" in test_class_name:
        tier = "Tier 1 (Feature Coverage)"
    else:
        tier = "Tier 1 (Feature Coverage)"

    # Determine Feature
    feature = "Multi-Feature / System"
    for feat_name, prefixes in FEATURE_MAPPING.items():
        if any(p in test_name for p in prefixes):
            feature = feat_name
            break

    return tier, feature


def main():
    print("=" * 80)
    print("  MyoPS-Net MyoPS380 Adaptation — Master E2E Test Suite")
    print("  Protocol: myops380_train304_test76_last_v1 | Metric: myops380_voxel_v3")
    print("=" * 80)

    loader = unittest.TestLoader()
    master_suite = unittest.TestSuite()

    all_tests_metadata = []

    for mod_name in TEST_MODULES:
        try:
            suite = loader.loadTestsFromName(mod_name)
            master_suite.addTests(suite)

            # Extract test names and classes
            def extract_tests(s):
                for item in s:
                    if isinstance(item, unittest.TestSuite):
                        extract_tests(item)
                    else:
                        t_name = item._testMethodName
                        c_name = item.__class__.__name__
                        tier, feat = categorize_test(item, t_name, c_name)
                        all_tests_metadata.append({
                            "name": t_name,
                            "class": c_name,
                            "module": mod_name,
                            "tier": tier,
                            "feature": feat,
                        })
            extract_tests(suite)
        except Exception as e:
            print(f"Error loading module {mod_name}: {e}")
            return 1

    total_count = master_suite.countTestCases()
    print(f"\nDiscovered {total_count} test cases across {len(TEST_MODULES)} modules.")
    print("Executing tests...\n")

    start_time = time.time()
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(master_suite)
    elapsed_time = time.time() - start_time

    # Tally metrics
    tier_counts = {
        "Tier 1 (Feature Coverage)": 0,
        "Tier 2 (Boundary & Corner Cases)": 0,
        "Tier 3 (Cross-Feature Combinations)": 0,
        "Tier 4 (Real-World Application Scenarios)": 0,
    }
    feature_counts = {f: 0 for f in FEATURE_MAPPING}
    feature_counts["Multi-Feature / System"] = 0

    for meta in all_tests_metadata:
        tier_counts[meta["tier"]] = tier_counts.get(meta["tier"], 0) + 1
        feature_counts[meta["feature"]] = feature_counts.get(meta["feature"], 0) + 1

    print("\n" + "=" * 80)
    print("                       TEST EXECUTION SUMMARY REPORT")
    print("=" * 80)
    print(f"Total Tests Executed : {result.testsRun}")
    print(f"Passed               : {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures             : {len(result.failures)}")
    print(f"Errors               : {len(result.errors)}")
    print(f"Skipped              : {len(result.skipped)}")
    print(f"Execution Duration   : {elapsed_time:.2f} seconds")
    print("-" * 80)

    print("\n[ Tier Breakdown ]")
    for tier_name, count in tier_counts.items():
        status = "PASSED" if len(result.failures) == 0 and len(result.errors) == 0 else "FAILED"
        print(f"  {tier_name:<42} : {count:>3} tests [{status}]")

    print("\n[ Feature Breakdown (F1 - F11) ]")
    for feat_name, count in sorted(feature_counts.items()):
        if count > 0:
            print(f"  {feat_name:<42} : {count:>3} tests")

    print("=" * 80)

    if result.wasSuccessful():
        print(">>> ALL TESTS PASSED SUCCESSFULLY (Exit Code 0) <<<")
        return 0
    else:
        print(">>> TEST SUITE REPORTED FAILURES (Exit Code 1) <<<")
        return 1


if __name__ == "__main__":
    sys.exit(main())
