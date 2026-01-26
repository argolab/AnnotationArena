#!/usr/bin/env python3
"""
Integration test for the renovated DataConverter.
Tests the complete workflow with a real bundle file.
"""

import json
import tempfile
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data import DataConverter, RankingData
from stan.pipeline.bundle import GroundTruthBundle


def create_realistic_bundle_data():
    """Create realistic bundle data similar to the actual data_bundle.json."""
    return {
        "embeddings": [
            [0.95863895, 0.071792844, -0.85574897, 0.24455114],
            [-2.4594651, -0.97315924, -0.65199801, -0.10439999],
            [-1.0699457, -0.30663965, 2.2796311, -1.1178155],
            [0.17414303, -0.82222609, -0.69397331, -1.1386762],
            [0.72248339, 1.1292263, 0.49864317, 0.63990686]
        ],
        "mean_preferences": [
            [0.58922607, -0.17710804, 0.57601068, 0.13962864],
            [-0.66525788, -0.32268889, 0.15108968, 2.0549043],
            [-0.15418018, -0.51953543, 0.15746221, 0.43066455]
        ],
        "annotator_preferences": [
            [0.28364953, -0.16153353, 0.77378189, 0.41757222],
            [0.63411204, -0.042241264, 0.92808711, -0.053587092],
            [0.2233171, -0.087471525, 0.65208065, 0.26398627],
            [1.0946333, -0.38039185, 0.057077267, 0.36693102],
            [0.55742358, -0.57170732, 0.46595295, 0.092380428],
            [0.85594024, -0.87924488, 0.46606754, -0.20804417]
        ],
        "rating_probs": [
            [0.043297059, 0.67299395, 0.028576804, 0.11353841, 0.14159377],
            [7.6076731e-05, 0.9589829, 0.00053367596, 0.0013180661, 0.039089277],
            [0.19569065, 0.17416806, 0.14329358, 0.01120437, 0.47564334],
            [0.97103822, 0.0015721769, 0.00096843563, 0.023721298, 0.0026998693],
            [0.26132357, 0.2045651, 0.53027767, 0.0032955434, 0.00053811922],
            [0.64913723, 0.13462898, 0.12133625, 0.032498251, 0.062399297]
        ],
        "rating_thresholds": [
            [0.043297059, 0.71629101, 0.74486781, 0.85840623, 1.0],
            [7.6076731e-05, 0.95905898, 0.95959266, 0.96091072, 1.0],
            [0.19569065, 0.36985871, 0.51315229, 0.52435666, 1.0],
            [0.97103822, 0.9726104, 0.97357883, 0.99730013, 1.0],
            [0.26132357, 0.46588866, 0.99616634, 0.99946188, 1.0],
            [0.64913723, 0.78376621, 0.90510245, 0.9376007, 1.0]
        ],
        "base_scores": [
            [3.841163, 2.1865048, 1.9662285, 1.4309401, -1.4034085],
            [5.3633606, 3.266741, 1.9730185, 1.8570572, -0.52374545],
            [4.130448, 2.4630603, 2.2371907, 0.48840193, -3.5267458],
            [5.4204434, 0.36305696, -0.41791388, 1.1873479, -2.2941495],
            [5.4636593, 3.0639882, 1.4627924, 2.8044889, -2.0566264],
            [5.2826587, 3.8618495, 1.6157459, 3.0135257, -3.1067159]
        ],
        "all_ratings": [
            {"attribute": 1, "annotator": 1, "item": 1, "value": 5, "instance": "train"},
            {"attribute": 1, "annotator": 1, "item": 2, "value": 4, "instance": "train"},
            {"attribute": 1, "annotator": 2, "item": 1, "value": 2, "instance": "train"},
            {"attribute": 1, "annotator": 2, "item": 2, "value": 2, "instance": "train"},
            {"attribute": 1, "annotator": 3, "item": 1, "value": 5, "instance": "train"},
            {"attribute": 1, "annotator": 3, "item": 2, "value": 5, "instance": "train"},
            {"attribute": 1, "annotator": 1, "item": 3, "value": 5, "instance": "test"},
            {"attribute": 1, "annotator": 1, "item": 4, "value": 2, "instance": "test"},
            {"attribute": 1, "annotator": 2, "item": 3, "value": 2, "instance": "test"},
            {"attribute": 1, "annotator": 2, "item": 4, "value": 2, "instance": "test"},
        ],
        "all_pairwise": [
            {"attribute": 1, "annotator": 1, "items": [1, 2], "order": [1, 2], "tied_rating": 4, "instance": "train"},
            {"attribute": 1, "annotator": 2, "items": [1, 2], "order": [1, 2], "tied_rating": 2, "instance": "train"},
            {"attribute": 1, "annotator": 3, "items": [1, 2], "order": [1, 2], "tied_rating": 5, "instance": "train"},
            {"attribute": 1, "annotator": 1, "items": [3, 4], "order": [1, 2], "tied_rating": 5, "instance": "test"},
            {"attribute": 1, "annotator": 2, "items": [3, 4], "order": [1, 2], "tied_rating": 2, "instance": "test"},
        ],
        "observed_ratings": [
            {"attribute": 1, "annotator": 1, "item": 1, "value": 5, "instance": "train"},
            {"attribute": 1, "annotator": 2, "item": 1, "value": 2, "instance": "train"},
            {"attribute": 1, "annotator": 3, "item": 1, "value": 5, "instance": "train"},
            {"attribute": 1, "annotator": 1, "item": 3, "value": 5, "instance": "test"},
            {"attribute": 1, "annotator": 2, "item": 3, "value": 2, "instance": "test"},
        ],
        "missing_ratings": [
            {"attribute": 1, "annotator": 1, "item": 2, "value": 4, "instance": "train"},
            {"attribute": 1, "annotator": 2, "item": 2, "value": 2, "instance": "train"},
            {"attribute": 1, "annotator": 3, "item": 2, "value": 5, "instance": "train"},
            {"attribute": 1, "annotator": 1, "item": 4, "value": 2, "instance": "test"},
            {"attribute": 1, "annotator": 2, "item": 4, "value": 2, "instance": "test"},
        ],
        "observed_pairwise": [
            {"attribute": 1, "annotator": 1, "items": [1, 2], "order": [1, 2], "tied_rating": 4, "instance": "train"},
            {"attribute": 1, "annotator": 2, "items": [1, 2], "order": [1, 2], "tied_rating": 2, "instance": "train"},
            {"attribute": 1, "annotator": 1, "items": [3, 4], "order": [1, 2], "tied_rating": 5, "instance": "test"},
        ],
        "missing_pairwise": [
            {"attribute": 1, "annotator": 3, "items": [1, 2], "order": [1, 2], "tied_rating": 5, "instance": "train"},
            {"attribute": 1, "annotator": 2, "items": [3, 4], "order": [1, 2], "tied_rating": 2, "instance": "test"},
        ],
        "stats": {
            "K_train": 2,
            "K_test": 2,
            "total_items": 4,
            "total_possible_ratings": 20,
            "total_ratings": 10,
            "observed_ratings": 5,
            "missing_ratings": 5,
            "train_ratings": 6,
            "test_ratings": 4,
            "train_observed": 3,
            "test_observed": 2,
            "total_pairwise": 5,
            "observed_pairwise": 3,
            "missing_pairwise": 2,
            "train_pairwise": 3,
            "test_pairwise": 2,
            "observation_rate": 0.5,
            "train_observation_rate": 0.5,
            "test_observation_rate": 0.5
        }
    }


def test_complete_workflow():
    """Test the complete workflow from bundle loading to variable creation."""
    print("Testing complete DataConverter workflow...")
    
    # Create converter
    converter = DataConverter(num_items=4, num_annotators=3, num_likert_classes=5, max_rank_size=2)
    
    # Create temporary bundle file
    sample_data = create_realistic_bundle_data()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_data, f)
        temp_file = f.name
    
    try:
        # Load bundle
        bundle = converter.load_bundle_data(temp_file)
        print(f"✓ Loaded bundle with {len(bundle.all_ratings)} ratings and {len(bundle.all_pairwise)} pairwise rankings")
        
        # Validate bundle
        errors = converter.validate_bundle(bundle)
        if errors:
            print(f"✗ Validation failed with errors: {errors}")
            return False
        print("✓ Bundle validation passed")
        
        # Test all four combinations
        train_obs = converter.create_variables_from_bundle(bundle, "train", "observed")
        test_obs = converter.create_variables_from_bundle(bundle, "test", "observed")
        train_missing = converter.create_variables_from_bundle(bundle, "train", "missing")
        test_missing = converter.create_variables_from_bundle(bundle, "test", "missing")
        
        print(f"✓ Created variables:")
        print(f"  - Train observed: {len(train_obs)}")
        print(f"  - Test observed: {len(test_obs)}")
        print(f"  - Train missing: {len(train_missing)}")
        print(f"  - Test missing: {len(test_missing)}")
        
        # Verify status properties
        assert all(v.is_observed for v in train_obs), "Train observed variables should have status=2"
        assert all(v.is_observed for v in test_obs), "Test observed variables should have status=2"
        assert all(v.is_missing for v in train_missing), "Train missing variables should have status=0"
        assert all(v.is_missing for v in test_missing), "Test missing variables should have status=0"
        
        # Verify instance filtering
        assert all(v.instance == "train" for v in train_obs), "Train observed should be train instance"
        assert all(v.instance == "test" for v in test_obs), "Test observed should be test instance"
        assert all(v.instance == "train" for v in train_missing), "Train missing should be train instance"
        assert all(v.instance == "test" for v in test_missing), "Test missing should be test instance"
        
        print("✓ All status and instance properties verified")
        
        # Test error handling
        try:
            converter.create_variables_from_bundle(bundle, "invalid", "observed")
            print("✗ Should have raised ValueError for invalid partition")
            return False
        except ValueError as e:
            if "Invalid partition: invalid" in str(e):
                print("✓ Error handling for invalid partition works")
            else:
                print(f"✗ Unexpected error message: {e}")
                return False
        
        try:
            converter.create_variables_from_bundle(bundle, "train", "invalid")
            print("✗ Should have raised ValueError for invalid status")
            return False
        except ValueError as e:
            if "Invalid status: invalid" in str(e):
                print("✓ Error handling for invalid status works")
            else:
                print(f"✗ Unexpected error message: {e}")
                return False
        
        # Test that 'all' options are rejected
        try:
            converter.create_variables_from_bundle(bundle, "all", "observed")
            print("✗ Should have raised ValueError for 'all' partition")
            return False
        except ValueError as e:
            if "Invalid partition: all" in str(e):
                print("✓ Error handling for 'all' partition works")
            else:
                print(f"✗ Unexpected error message: {e}")
                return False
        
        try:
            converter.create_variables_from_bundle(bundle, "train", "all")
            print("✗ Should have raised ValueError for 'all' status")
            return False
        except ValueError as e:
            if "Invalid status: all" in str(e):
                print("✓ Error handling for 'all' status works")
            else:
                print(f"✗ Unexpected error message: {e}")
                return False
        
        print("\n🎉 All tests passed! DataConverter renovation is working correctly.")
        return True
        
    finally:
        os.unlink(temp_file)


if __name__ == "__main__":
    success = test_complete_workflow()
    exit(0 if success else 1)
