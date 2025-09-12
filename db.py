import sqlite3

def get_connection():
    return sqlite3.connect("rugby.db")

def get_players():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT name FROM players")
    players = [row[0] for row in cur.fetchall()]
    conn.close()
    return players

def get_clubs():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT name FROM clubs")
    clubs = [row[0] for row in cur.fetchall()]
    conn.close()
    return clubs
