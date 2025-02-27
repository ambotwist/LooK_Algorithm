"""
Initialization utilities for the recommender system.
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client

def init_supabase() -> Client:
    """
    Initialize the Supabase client using environment variables.
    
    Returns:
        Initialized Supabase client
        
    Raises:
        ValueError: If required environment variables are not set
    """
    # Load environment variables
    load_dotenv()
    
    # Get Supabase credentials
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables must be set.")
        
    # Create and return Supabase client
    return create_client(supabase_url, supabase_key) 