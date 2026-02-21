import sqlite3
from pathlib import Path

DB_PATH = Path("data/contributions.db")

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Create contributions table
    c.execute('''
        CREATE TABLE IF NOT EXISTS contributions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            location TEXT,
            verified INTEGER DEFAULT 0,
            source_ip TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Initialized database at {DB_PATH}")

if __name__ == "__main__":
    init_db()
