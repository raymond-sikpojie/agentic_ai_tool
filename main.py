import sqlite3

# Connect to the database
conn = sqlite3.connect("music.db")
cursor = conn.cursor()

# Run a query
cursor.execute("SELECT * FROM music WHERE artist = 'The Beatles'")
rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()
