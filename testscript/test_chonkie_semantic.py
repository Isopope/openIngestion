import sys
from typing import Sequence
import numpy as np

from openingestion.document import ContentBlock, BlockKind
from openingestion.chunker.by_semantic import SemanticChunker

class MockEmbedder:
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Return 10-dimensional random vectors for each sentence to avoid needing sentence-transformers in the venv
        return [np.random.rand(10).tolist() for _ in texts]

def test_semantic_chunker():
    blocks = [
        ContentBlock(
            kind=BlockKind.TEXT,
            text="Hello world. The weather is nice. " * 3,
            page_idx=0,
            block_index=0,
            reading_order=0,
            bbox=(0, 0, 100, 100)
        ),
        ContentBlock(
            kind=BlockKind.TEXT,
            text="Dogs are great pets. Cats are also very cool. Birds are amazing animals. " * 3,
            page_idx=0,
            block_index=1,
            reading_order=1,
            bbox=(0, 100, 100, 200)
        )
    ]
    
    # Test Percentile Mode
    print("Testing Percentile Breakpoints...")
    chunker1 = SemanticChunker(
        model=MockEmbedder(),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_value=50.0,
        sg_window_length=3,
        sg_polyorder=2
    )
    chunks1 = chunker1.chunk(blocks, source="test1.pdf")
    
    # Test Fixed Threshold Mode
    print("Testing Threshold Breakpoints...")
    chunker2 = SemanticChunker(
        model=MockEmbedder(),
        breakpoint_threshold_type="threshold",
        breakpoint_threshold_value=0.5,
        sg_window_length=3,
        sg_polyorder=2
    )
    chunks2 = chunker2.chunk(blocks, source="test2.pdf")
    
    print("SemanticChunker verification completed successfully.")

if __name__ == "__main__":
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    test_semantic_chunker()
