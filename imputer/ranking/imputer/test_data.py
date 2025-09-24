import pytest
from dataclasses import dataclass
from typing import List, Optional
from data import RankingData


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


if __name__ == "__main__":
    pytest.main([__file__])
