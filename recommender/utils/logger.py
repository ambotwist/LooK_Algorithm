"""
Logger utility for the recommender system.
"""

import logging

def setup_logger(name="hm_recommender") -> logging.Logger:
    """
    Set up a logger for the recommendation system.
    
    Args:
        name: Name of the logger
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Check if the logger already has handlers to avoid duplicates
    if not logger.handlers:
        # Create console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger 