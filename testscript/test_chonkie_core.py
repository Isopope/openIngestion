import sys
import logging
from loguru import logger
from openingestion.document import ContentBlock, BlockKind
from openingestion.chunker.by_sentence import SentenceChunker

def test_chunker_fallback():
    # Setup test blocks
    text = "Hello world. This is a simple test! How are you doing? I am fine, thank you."
    block = ContentBlock(
        kind=BlockKind.TEXT,
        text=text,
        page_idx=0,
        block_index=0,
        reading_order=0,
        bbox=(0, 0, 100, 100)
    )

    chunker = SentenceChunker(
        chunk_size=10, 
        chunk_overlap=0, 
        min_sentences_per_chunk=1, 
        tokenizer="word"  # Word tokenizer for deterministic counts
    )

    print("Running SentenceChunker Without chonkie-core (fallback):")
    chunks = chunker.chunk([block], source="test_doc.pdf")
    for i, chunk in enumerate(chunks):
        print(f" Chunk {i}: {repr(chunk.page_content)}")
    print("Fallback successful\\n")

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    test_chunker_fallback()
