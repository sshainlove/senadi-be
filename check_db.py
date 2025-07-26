from app import get_db_connection
import json

def check_tables():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Cek struktur tabel messages
    print("=== STRUKTUR TABEL MESSAGES ===")
    cursor.execute("DESCRIBE messages")
    columns = cursor.fetchall()
    for col in columns:
        print(f"{col['Field']}: {col['Type']} (Null: {col['Null']}, Key: {col['Key']}, Default: {col['Default']}, Extra: {col['Extra']})")
    
    # Cek contoh data
    print("\n=== CONTOH DATA PESAN ===")
    cursor.execute("SELECT * FROM messages LIMIT 5")
    messages = cursor.fetchall()
    for msg in messages:
        print(json.dumps(msg, default=str, indent=2))
    
    # Cek jumlah pesan per chat
    print("\n=== JUMLAH PESAN PER CHAT ===")
    cursor.execute("SELECT chat_id, COUNT(*) as count FROM messages GROUP BY chat_id ORDER BY count DESC LIMIT 10")
    counts = cursor.fetchall()
    for count in counts:
        print(f"Chat {count['chat_id']}: {count['count']} pesan")
    
    # Cek jumlah total pesan
    cursor.execute("SELECT COUNT(*) as total FROM messages")
    total = cursor.fetchone()['total']
    print(f"\nTotal pesan dalam database: {total}")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    check_tables() 