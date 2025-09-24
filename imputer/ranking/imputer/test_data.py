import pytest
import json
import tempfile
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dataclasses import dataclass
from typing import List, Optional
from data import RankingData, DataConverter
from stan.pipeline.bundle import GroundTruthBundle


class TestRankingData:
    """Test cases for RankingData class."""
    
    def test_rating_data_creation(self):
        """Test creating a rating RankingData object."""
        rating = RankingData(
            annotator_id=0,
            attribute_id=1,
            is_listwise=False,
            item_ids=[2],
            status=2,
            instance="train",
            rating_value=3
        )
        
        assert rating.annotator_id == 0
        assert rating.attribute_id == 1
        assert not rating.is_listwise
        assert rating.item_ids == [2]
        assert rating.status == 2
        assert rating.instance == "train"
        assert rating.rating_value == 3
        assert rating.ranking_order is None
    
    def test_pairwise_data_creation(self):
        """Test creating a pairwise ranking RankingData object."""
        pairwise = RankingData(
            annotator_id=1,
            attribute_id=0,
            is_listwise=True,
            item_ids=[1, 3],
            status=0,
            instance="test",
            ranking_order=[1, 2]
        )
        
        assert pairwise.annotator_id == 1
        assert pairwise.attribute_id == 0
        assert pairwise.is_listwise
        assert pairwise.item_ids == [1, 3]
        assert pairwise.status == 0
        assert pairwise.instance == "test"
        assert pairwise.ranking_order == [1, 2]
        assert pairwise.rating_value is None
    
    def test_status_properties_missing(self):
        """Test status properties for missing data (status=0)."""
        data = RankingData(
            annotator_id=0,
            attribute_id=0,
            is_listwise=False,
            item_ids=[0],
            status=0,
            instance="train"
        )
        
        assert data.is_missing
        assert not data.is_masked
        assert not data.is_observed
    
    def test_status_properties_masked(self):
        """Test status properties for masked data (status=1)."""
        data = RankingData(
            annotator_id=0,
            attribute_id=0,
            is_listwise=False,
            item_ids=[0],
            status=1,
            instance="train"
        )
        
        assert not data.is_missing
        assert data.is_masked
        assert not data.is_observed
    
    def test_status_properties_observed(self):
        """Test status properties for observed data (status=2)."""
        data = RankingData(
            annotator_id=0,
            attribute_id=0,
            is_listwise=False,
            item_ids=[0],
            status=2,
            instance="train"
        )
        
        assert not data.is_missing
        assert not data.is_masked
        assert data.is_observed
    
    def test_invalid_status(self):
        """Test that invalid status codes work but properties return False."""
        data = RankingData(
            annotator_id=0,
            attribute_id=0,
            is_listwise=False,
            item_ids=[0],
            status=5,  # Invalid status
            instance="train"
        )
        
        assert not data.is_missing
        assert not data.is_masked
        assert not data.is_observed


class TestDataConverter:
    """Test cases for DataConverter class."""
    
    def create_sample_bundle_data(self):
        """Create sample bundle data for testing."""
        return {
            "embeddings": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            "mean_preferences": [[0.1, 0.2], [0.3, 0.4]],
            "annotator_preferences": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
            "rating_probs": [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.1]],
            "rating_thresholds": [[0.1, 0.3, 0.6, 1.0], [0.2, 0.4, 0.7, 1.0]],
            "base_scores": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "all_ratings": [
                {"attribute": 1, "annotator": 1, "item": 1, "value": 3, "instance": "train"},
                {"attribute": 1, "annotator": 1, "item": 2, "value": 2, "instance": "train"},
                {"attribute": 1, "annotator": 2, "item": 1, "value": 4, "instance": "test"},
            ],
            "all_pairwise": [
                {"attribute": 1, "annotator": 1, "items": [1, 2], "order": [1, 2], "tied_rating": 3, "instance": "train"},
                {"attribute": 1, "annotator": 2, "items": [2, 1], "order": [2, 1], "tied_rating": 2, "instance": "test"},
            ],
            "observed_ratings": [
                {"attribute": 1, "annotator": 1, "item": 1, "value": 3, "instance": "train"},
            ],
            "missing_ratings": [
                {"attribute": 1, "annotator": 1, "item": 2, "value": 2, "instance": "train"},
                {"attribute": 1, "annotator": 2, "item": 1, "value": 4, "instance": "test"},
            ],
            "observed_pairwise": [
                {"attribute": 1, "annotator": 1, "items": [1, 2], "order": [1, 2], "tied_rating": 3, "instance": "train"},
            ],
            "missing_pairwise": [
                {"attribute": 1, "annotator": 2, "items": [2, 1], "order": [2, 1], "tied_rating": 2, "instance": "test"},
            ],
            "stats": {"total_items": 3, "total_ratings": 3}
        }
    
    def test_load_bundle_data(self):
        """Test loading bundle data from JSON file."""
        converter = DataConverter(num_items=3, num_likert_classes=5)
        
        # Create temporary file with sample data
        sample_data = self.create_sample_bundle_data()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_data, f)
            temp_file = f.name
        
        try:
            bundle = converter.load_bundle_data(temp_file)
            
            assert isinstance(bundle, GroundTruthBundle)
            assert bundle.embeddings.shape == (3, 2)
            assert len(bundle.all_ratings) == 3
            assert len(bundle.all_pairwise) == 2
            assert len(bundle.observed_ratings) == 1
            assert len(bundle.missing_ratings) == 2
            
        finally:
            os.unlink(temp_file)
    
    def test_create_variables_from_bundle_train_observed(self):
        """Test creating variables from bundle with train observed status."""
        converter = DataConverter(num_items=3, num_likert_classes=5)
        sample_data = self.create_sample_bundle_data()
        bundle = GroundTruthBundle.from_dict(sample_data)
        
        variables = converter.create_variables_from_bundle(bundle, "train", "observed")
        
        assert len(variables) == 2  # 1 rating + 1 pairwise
        assert all(v.is_observed for v in variables)
        assert all(v.instance == "train" for v in variables)
        
        # Check rating variable
        rating_vars = [v for v in variables if not v.is_listwise]
        assert len(rating_vars) == 1
        rating = rating_vars[0]
        assert rating.annotator_id == 0
        assert rating.attribute_id == 0
        assert rating.item_ids == [0]
        assert rating.rating_value == 2  # 3-1
        assert rating.instance == "train"
        
        # Check pairwise variable
        pairwise_vars = [v for v in variables if v.is_listwise]
        assert len(pairwise_vars) == 1
        pairwise = pairwise_vars[0]
        assert pairwise.annotator_id == 0
        assert pairwise.attribute_id == 0
        assert pairwise.item_ids == [0, 1]
        assert pairwise.ranking_order == [1, 2]
        assert pairwise.instance == "train"
    
    def test_create_variables_from_bundle_train_missing(self):
        """Test creating variables from bundle with train missing status."""
        converter = DataConverter(num_items=3, num_likert_classes=5)
        sample_data = self.create_sample_bundle_data()
        bundle = GroundTruthBundle.from_dict(sample_data)
        
        variables = converter.create_variables_from_bundle(bundle, "train", "missing")
        
        assert len(variables) == 1  # 1 rating
        assert all(v.is_missing for v in variables)
        assert all(v.instance == "train" for v in variables)
    
    def test_create_variables_from_bundle_test_missing(self):
        """Test creating variables from bundle with test missing status."""
        converter = DataConverter(num_items=3, num_likert_classes=5)
        sample_data = self.create_sample_bundle_data()
        bundle = GroundTruthBundle.from_dict(sample_data)
        
        variables = converter.create_variables_from_bundle(bundle, "test", "missing")
        
        assert len(variables) == 2  # 1 rating + 1 pairwise
        assert all(v.is_missing for v in variables)
        assert all(v.instance == "test" for v in variables)
    
    def test_create_variables_from_bundle_test_observed(self):
        """Test creating variables from bundle with test observed status."""
        converter = DataConverter(num_items=3, num_likert_classes=5)
        sample_data = self.create_sample_bundle_data()
        bundle = GroundTruthBundle.from_dict(sample_data)
        
        variables = converter.create_variables_from_bundle(bundle, "test", "observed")
        
        assert len(variables) == 0  # No observed test data in sample
        assert all(v.is_observed for v in variables)
        assert all(v.instance == "test" for v in variables)
    
    def test_create_variables_from_bundle_invalid_status(self):
        """Test error handling for invalid status."""
        converter = DataConverter(num_items=3, num_likert_classes=5)
        sample_data = self.create_sample_bundle_data()
        bundle = GroundTruthBundle.from_dict(sample_data)
        
        with pytest.raises(ValueError, match="Invalid status: invalid"):
            converter.create_variables_from_bundle(bundle, "train", "invalid")
    
    def test_create_variables_from_bundle_invalid_partition(self):
        """Test error handling for invalid partition."""
        converter = DataConverter(num_items=3, num_likert_classes=5)
        sample_data = self.create_sample_bundle_data()
        bundle = GroundTruthBundle.from_dict(sample_data)
        
        with pytest.raises(ValueError, match="Invalid partition: invalid"):
            converter.create_variables_from_bundle(bundle, "invalid", "observed")
    
    def test_create_variables_from_bundle_all_status_rejected(self):
        """Test that 'all' status is rejected."""
        converter = DataConverter(num_items=3, num_likert_classes=5)
        sample_data = self.create_sample_bundle_data()
        bundle = GroundTruthBundle.from_dict(sample_data)
        
        with pytest.raises(ValueError, match="Invalid status: all"):
            converter.create_variables_from_bundle(bundle, "train", "all")
    
    def test_create_variables_from_bundle_all_partition_rejected(self):
        """Test that 'all' partition is rejected."""
        converter = DataConverter(num_items=3, num_likert_classes=5)
        sample_data = self.create_sample_bundle_data()
        bundle = GroundTruthBundle.from_dict(sample_data)
        
        with pytest.raises(ValueError, match="Invalid partition: all"):
            converter.create_variables_from_bundle(bundle, "all", "observed")
    
    def test_all_combinations(self):
        """Test all valid combinations of partition and status."""
        converter = DataConverter(num_items=3, num_likert_classes=5)
        sample_data = self.create_sample_bundle_data()
        bundle = GroundTruthBundle.from_dict(sample_data)
        
        # Test all four combinations
        train_obs = converter.create_variables_from_bundle(bundle, "train", "observed")
        train_missing = converter.create_variables_from_bundle(bundle, "train", "missing")
        test_obs = converter.create_variables_from_bundle(bundle, "test", "observed")
        test_missing = converter.create_variables_from_bundle(bundle, "test", "missing")
        
        # Verify counts
        assert len(train_obs) == 2  # 1 rating + 1 pairwise
        assert len(train_missing) == 1  # 1 rating
        assert len(test_obs) == 0  # No observed test data in sample
        assert len(test_missing) == 2  # 1 rating + 1 pairwise
        
        # Verify status properties
        assert all(v.is_observed for v in train_obs)
        assert all(v.is_missing for v in train_missing)
        assert all(v.is_observed for v in test_obs)
        assert all(v.is_missing for v in test_missing)
        
        # Verify instance properties
        assert all(v.instance == "train" for v in train_obs)
        assert all(v.instance == "train" for v in train_missing)
        assert all(v.instance == "test" for v in test_obs)
        assert all(v.instance == "test" for v in test_missing)
    
    def test_validate_bundle_valid_data(self):
        """Test validation with valid bundle data."""
        converter = DataConverter(num_items=3, num_likert_classes=5)
        sample_data = self.create_sample_bundle_data()
        bundle = GroundTruthBundle.from_dict(sample_data)
        
        errors = converter.validate_bundle(bundle)
        assert len(errors) == 0, f"Validation failed with errors: {errors}"
    
    def test_validate_bundle_invalid_rating_value(self):
        """Test validation with invalid rating values."""
        converter = DataConverter(num_items=3, num_likert_classes=5)
        sample_data = self.create_sample_bundle_data()
        
        # Add invalid rating value
        sample_data["all_ratings"].append({
            "attribute": 1, "annotator": 1, "item": 1, "value": 10, "instance": "train"
        })
        
        bundle = GroundTruthBundle.from_dict(sample_data)
        errors = converter.validate_bundle(bundle)
        
        assert len(errors) > 0
        assert any("Invalid rating value 10" in error for error in errors)
    
    def test_validate_bundle_invalid_item_index(self):
        """Test validation with invalid item indices."""
        converter = DataConverter(num_items=3, num_likert_classes=5)
        sample_data = self.create_sample_bundle_data()
        
        # Add invalid item index
        sample_data["all_ratings"].append({
            "attribute": 1, "annotator": 1, "item": 10, "value": 3, "instance": "train"
        })
        
        bundle = GroundTruthBundle.from_dict(sample_data)
        errors = converter.validate_bundle(bundle)
        
        assert len(errors) > 0
        assert any("Invalid item index 10" in error for error in errors)
    
    def test_validate_bundle_invalid_pairwise_items(self):
        """Test validation with invalid pairwise ranking items."""
        converter = DataConverter(num_items=3, num_likert_classes=5)
        sample_data = self.create_sample_bundle_data()
        
        # Add invalid pairwise ranking
        sample_data["all_pairwise"].append({
            "attribute": 1, "annotator": 1, "items": [1, 2, 3], "order": [1, 2, 3], 
            "tied_rating": 3, "instance": "train"
        })
        
        bundle = GroundTruthBundle.from_dict(sample_data)
        errors = converter.validate_bundle(bundle)
        
        assert len(errors) > 0
        assert any("Pairwise ranking must have exactly 2 items" in error for error in errors)
    
    def test_validate_bundle_partition_mismatch(self):
        """Test validation with incorrect observed/missing partition."""
        converter = DataConverter(num_items=3, num_likert_classes=5)
        sample_data = self.create_sample_bundle_data()
        
        # Add rating to observed that's not in all_ratings
        sample_data["observed_ratings"].append({
            "attribute": 2, "annotator": 1, "item": 1, "value": 3, "instance": "train"
        })
        
        bundle = GroundTruthBundle.from_dict(sample_data)
        errors = converter.validate_bundle(bundle)
        
        assert len(errors) > 0
        assert any("Observed and missing ratings do not partition all_ratings correctly" in error for error in errors)


if __name__ == "__main__":
    # Simple test runner
    print("Running DataConverter tests...")
    
    # Test RankingData
    print("\n=== Testing RankingData ===")
    test_ranking_data = TestRankingData()
    test_ranking_data.test_rating_data_creation()
    test_ranking_data.test_pairwise_data_creation()
    test_ranking_data.test_status_properties_missing()
    test_ranking_data.test_status_properties_masked()
    test_ranking_data.test_status_properties_observed()
    test_ranking_data.test_invalid_status()
    print("✓ RankingData tests passed")
    
    # Test DataConverter
    print("\n=== Testing DataConverter ===")
    test_converter = TestDataConverter()
    test_converter.test_load_bundle_data()
    test_converter.test_create_variables_from_bundle_train_observed()
    test_converter.test_create_variables_from_bundle_train_missing()
    test_converter.test_create_variables_from_bundle_test_missing()
    test_converter.test_create_variables_from_bundle_test_observed()
    test_converter.test_create_variables_from_bundle_invalid_status()
    test_converter.test_create_variables_from_bundle_invalid_partition()
    test_converter.test_create_variables_from_bundle_all_status_rejected()
    test_converter.test_create_variables_from_bundle_all_partition_rejected()
    test_converter.test_all_combinations()
    test_converter.test_validate_bundle_valid_data()
    test_converter.test_validate_bundle_invalid_rating_value()
    test_converter.test_validate_bundle_invalid_item_index()
    test_converter.test_validate_bundle_invalid_pairwise_items()
    test_converter.test_validate_bundle_partition_mismatch()
    print("✓ DataConverter tests passed")
    
    print("\n🎉 All tests passed!")
