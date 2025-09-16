"""Usage examples for LLM I/O models."""

from uuid import uuid4
from echo_roots.models import (
    CategoryClassificationInput,
    CategoryOption,
    CategoryClassificationOutput,
    Evidence,
    AttributeExtractionInput,
    AttributeOption,
    AttributeValueOption,
    AttributeExtractionOutput,
    AttributeSelection
)

def category_classification_example():
    """Category Classification Example."""
    
    # Input example
    category_input = CategoryClassificationInput(
        item_id=uuid4(),
        domain="product",
        title="iPhone 15 Pro 天然鈦色 256GB",
        description="Apple 最新旗艦手機",
        specs={"brand": "Apple", "storage": "256GB", "color": "天然鈦"},
        options=[
            CategoryOption(
                option_id=1,
                category_id=uuid4(),
                path=["Electronics", "Phones", "Smartphones"],
                labels={"zh-TW": "智慧手機", "en": "Smartphones"}
            ),
            CategoryOption(
                option_id=2, 
                category_id=uuid4(),
                path=["Electronics", "Accessories", "Cases"],
                labels={"zh-TW": "手機殼", "en": "Phone Cases"}
            )
        ]
    )

    # Output example
    category_output = CategoryClassificationOutput(
        selected_option=1,
        confidence=0.95,
        evidence=[
            Evidence(
                source="title",
                span=(0, 12),
                text="iPhone 15 Pro",
                score=0.95,
                model_id="gpt-4o-mini"
            )
        ],
        reasoning="Based on the product name 'iPhone 15 Pro', this is clearly a smartphone."
    )
    
    return category_input, category_output


def attribute_extraction_example():
    """Attribute Extraction Example."""
    
    # Input example
    attr_input = AttributeExtractionInput(
        item_id=uuid4(),
        domain="product",
        title="iPhone 15 Pro 天然鈦色 256GB",
        category_id=uuid4(),
        category_path=["Electronics", "Phones", "Smartphones"],
        attributes={
            "A1": AttributeOption(
                attr_key="A1",
                attr_id=uuid4(),
                attr_name="color",
                attr_labels={"zh-TW": "顏色", "en": "Color"},
                values={
                    "1": AttributeValueOption(
                        option_id=1,
                        value_id=uuid4(),
                        value="天然鈦",
                        aliases=["天然鈦色", "鈦色"]
                    ),
                    "2": AttributeValueOption(
                        option_id=2,
                        value_id=uuid4(), 
                        value="黑色",
                        aliases=["純黑色", "深空黑"]
                    )
                }
            ),
            "A2": AttributeOption(
                attr_key="A2",
                attr_id=uuid4(),
                attr_name="storage",
                attr_labels={"zh-TW": "儲存容量", "en": "Storage"},
                values={
                    "1": AttributeValueOption(
                        option_id=1,
                        value_id=uuid4(),
                        value="256GB"
                    )
                }
            )
        }
    )

    # Output example
    attr_output = AttributeExtractionOutput(
        selected_options={
            "A1": AttributeSelection(
                selected_options=[1],
                confidence=0.92,
                evidence=[
                    Evidence(
                        source="title",
                        span=(16, 19),
                        text="天然鈦色",
                        score=0.92,
                        model_id="gpt-4o-mini"
                    )
                ]
            ),
            "A2": AttributeSelection(
                selected_options=[1],
                confidence=0.98,
                evidence=[
                    Evidence(
                        source="title", 
                        span=(22, 27),
                        text="256GB",
                        score=0.98,
                        model_id="gpt-4o-mini"
                    )
                ]
            )
        },
        overall_confidence=0.95,
        reasoning="Color extracted from title '天然鈦色', storage capacity from '256GB'."
    )
    
    return attr_input, attr_output