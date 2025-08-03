#!/usr/bin/env python3
"""
Quick test of the fixed framework.
"""

from test_simple import run_simple_experiment

if __name__ == "__main__":
    print("Quick Framework Test")
    print("=" * 50)
    
    # Test 1: Single small experiment
    print("\n1. Testing single small experiment...")
    result1 = run_simple_experiment(
        n_nodes=5,
        train_size=100,
        target_parents=1.0,
        missing_rate=0.4,
        neural_type="structure"
    )
    
    if result1 and result1['status'] == 'SUCCESS':
        print("✅ Structure-only neural model test PASSED")
    else:
        print("❌ Structure-only neural model test FAILED")
    
    # Test 2: CPTs version  
    print("\n2. Testing CPTs neural model...")
    try:
        result2 = run_simple_experiment(
            n_nodes=5,
            train_size=100,
            target_parents=1.0,
            missing_rate=0.4,
            neural_type="cpts"
        )
        
        if result2 and result2['status'] == 'SUCCESS':
            print("✅ Structure+CPTs neural model test PASSED")
        else:
            print("❌ Structure+CPTs neural model test FAILED")
            print(f"   Result: {result2}")
    except Exception as e:
        print(f"❌ Structure+CPTs neural model test ERROR: {e}")
    
    print("\nQuick test completed!")