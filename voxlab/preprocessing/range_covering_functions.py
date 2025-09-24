"""
Range Covering Functions
Returns list of intervals representing chunk positions for covering a range [0, length)
"""

import math
from typing import List


def get_chunks_exactly(length: float, chunk_size: float, num_chunks: int) -> List[List[float]]:
    """
    Divide a range [0, length) into exactly num_chunks of given chunk_size.
    
    Chunks are evenly distributed with calculated spacing to ensure:
    - First chunk starts at 0
    - Last chunk ends at length
    - All intermediate chunks have equal spacing
    
    Args:
        length: Total length of the range to cover [0, length)
        chunk_size: Size of each chunk
        num_chunks: Exact number of chunks to create
    
    Returns:
        List of [start, end] intervals representing chunk positions
        
    Raises:
        ValueError: If num_chunks < 1 or if single chunk cannot cover the range
        
    Notes:
        - Spacing between chunks is calculated as: (length - num_chunks * chunk_size) / (num_chunks - 1)
        - Positive spacing means gaps between chunks
        - Negative spacing means chunks overlap
        - Zero spacing means chunks touch exactly
    """
    if num_chunks < 1:
        raise ValueError(f"Number of chunks must be at least 1, got {num_chunks}")
    
    # Special case: single chunk
    if num_chunks == 1:
        if abs(chunk_size - length) > 1e-10:
            raise ValueError(f"Single chunk of size {chunk_size} cannot cover range [0, {length})")
        return [[0, length]]
    
    # Calculate spacing between chunks
    # Formula: length = num_chunks * chunk_size + (num_chunks - 1) * spacing
    # Solving for spacing: spacing = (length - num_chunks * chunk_size) / (num_chunks - 1)
    spacing = (length - num_chunks * chunk_size) / (num_chunks - 1)
    
    # Generate chunk intervals
    intervals = []
    for i in range(num_chunks):
        start = i * (chunk_size + spacing)
        end = start + chunk_size
        intervals.append([start, end])
    
    # Adjust last chunk to end exactly at length (handle floating point precision)
    intervals[-1][1] = length
    
    return intervals


def get_chunks_max_spacing(length: float, chunk_size: float, max_spacing: float) -> List[List[float]]:
    """
    Find minimum number of chunks needed to cover [0, length) with spacing ≤ max_spacing.
    
    Minimizes the number of chunks while ensuring the spacing between consecutive
    chunks does not exceed max_spacing.
    
    Args:
        length: Total length of the range to cover [0, length)
        chunk_size: Size of each chunk
        max_spacing: Maximum allowed spacing between consecutive chunks
                    (can be negative to enforce minimum overlap)
    
    Returns:
        List of [start, end] intervals representing chunk positions
        
    Raises:
        ValueError: If constraints are violated:
            - max_spacing ≤ -chunk_size (would mean complete overlap or backwards)
            - max_spacing ≥ length - 2*chunk_size and chunk_size < length 
              (impossible to cover range with ≥2 chunks)
        
    Notes:
        - If max_spacing > 0: chunks may have gaps up to max_spacing
        - If max_spacing = 0: chunks will touch or overlap
        - If max_spacing < 0: chunks will overlap by at least |max_spacing|
        - Actual spacing may be less than max_spacing to ensure exact coverage
    """
    # Special case: single chunk can cover entire range
    if chunk_size >= length:
        return [[0, length]]
    
    # Validate spacing constraints
    if max_spacing <= -chunk_size:
        raise ValueError(
            f"max_spacing ({max_spacing}) must be greater than -chunk_size ({-chunk_size}). "
            "Chunks cannot overlap by more than their entire size."
        )
    
    # For ranges requiring multiple chunks, check if spacing is achievable
    # Only validate when length - 2*chunk_size is positive (non-overlapping case)
    if chunk_size < length and length - 2 * chunk_size > 0 and max_spacing >= length - 2 * chunk_size:
        raise ValueError(
            f"max_spacing ({max_spacing}) must be less than length - 2*chunk_size "
            f"({length - 2*chunk_size:.2f}) to cover the range with multiple chunks."
        )
    
    # Calculate minimum number of chunks needed
    # Each chunk covers chunk_size units, and we can have max_spacing gap between them
    # So each chunk effectively covers (chunk_size + max_spacing) units, except the last
    num_chunks = math.ceil((length + max_spacing) / (chunk_size + max_spacing))
    num_chunks = max(1, num_chunks)
    
    # Use get_chunks_exactly to generate the actual positions
    return get_chunks_exactly(length, chunk_size, num_chunks)


def get_chunks_min_spacing(length: float, chunk_size: float, min_spacing: float) -> List[List[float]]:
    """
    Find maximum number of chunks that can fit in [0, length) with spacing ≥ min_spacing.
    
    Maximizes the number of chunks while ensuring the spacing between consecutive
    chunks is at least min_spacing.
    
    Args:
        length: Total length of the range to cover [0, length)
        chunk_size: Size of each chunk
        min_spacing: Minimum required spacing between consecutive chunks
                    (can be negative to allow overlap up to |min_spacing|)
    
    Returns:
        List of [start, end] intervals representing chunk positions
        
    Raises:
        ValueError: If constraints are violated:
            - min_spacing ≤ -chunk_size (would mean complete overlap or backwards)
            - min_spacing ≥ length - 2*chunk_size and chunk_size < length 
              (would result in fewer than 2 chunks)
        
    Notes:
        - If min_spacing > 0: chunks will have gaps of at least min_spacing
        - If min_spacing = 0: chunks may touch but not overlap
        - If min_spacing < 0: chunks may overlap by up to |min_spacing|
        - Actual spacing may be greater than min_spacing to ensure exact coverage
    """
    # Special case: chunk larger than range
    if chunk_size >= length:
        return [[0, length]]
    
    # Validate spacing constraints
    if min_spacing <= -chunk_size:
        raise ValueError(
            f"min_spacing ({min_spacing}) must be greater than -chunk_size ({-chunk_size}). "
            "Chunks cannot overlap by more than their entire size."
        )
    
    # For ranges requiring multiple chunks, check if spacing makes sense
    # Only validate when length - 2*chunk_size is positive (non-overlapping case)
    if chunk_size < length and length - 2 * chunk_size > 0 and min_spacing >= length - 2 * chunk_size:
        raise ValueError(
            f"min_spacing ({min_spacing}) must be less than length - 2*chunk_size "
            f"({length - 2*chunk_size:.2f}) to allow for at least 2 chunks."
        )
    
    # Calculate maximum number of chunks that can fit
    # Each chunk needs at least (chunk_size + min_spacing) space, except the last
    num_chunks = math.floor((length + min_spacing) / (chunk_size + min_spacing))
    num_chunks = max(1, num_chunks)
    
    # Use get_chunks_exactly to generate the actual positions
    return get_chunks_exactly(length, chunk_size, num_chunks)