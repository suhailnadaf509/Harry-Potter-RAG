"""
This module overrides the system SQLite with a newer version from pysqlite3.
It must be imported before any other imports that might use SQLite.
"""
import sys
import logging

try:
    # Try to import pysqlite3 and replace the sqlite3 module in sys.modules
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
    logging.info("Successfully replaced sqlite3 with pysqlite3")
except ImportError:
    logging.warning("Could not import pysqlite3. Using system sqlite3 instead.")
    import sqlite3
    
    # Check SQLite version
    sqlite_version = sqlite3.sqlite_version_info
    logging.info(f"Using SQLite version: {sqlite_version}")
    
    if sqlite_version < (3, 35, 0):
        logging.warning("SQLite version is less than 3.35.0. Chroma may not work correctly.") 