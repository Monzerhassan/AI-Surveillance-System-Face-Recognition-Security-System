import sqlite3

def connect_db():
    conn = sqlite3.connect("database.db")
    return conn

def create_table():
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS persons (
        name TEXT PRIMARY KEY,
        age INTEGER,
        status TEXT,
        action TEXT
    )
    """)

    conn.commit()
    conn.close()

def get_person(name):
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM persons WHERE name=?", (name,))
    data = cursor.fetchone()

    conn.close()
    return data
