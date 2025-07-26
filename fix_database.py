import mysql.connector
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_db_connection():
    db_name = 'chatbot.susenas'  # Hardcoded from logs
    print(f"Connecting to database: {db_name}")
    return mysql.connector.connect(
        host='localhost',
        user='root', 
        password='',
        database=db_name
    )

def fix_messages_table():
    """Fix the messages table by changing its primary key to AUTO_INCREMENT INT"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        print("Checking messages table structure...")
        cursor.execute("SHOW CREATE TABLE messages")
        table_schema = cursor.fetchone()[1]
        print(f"Current table schema: {table_schema}")
        
        # Check if schema is already correct
        if 'AUTO_INCREMENT' in table_schema:
            print("✅ Table already has AUTO_INCREMENT - no need to fix.")
            cursor.close()
            conn.close()
            return True
        
        # Create a new table with the correct schema
        print("Creating new messages table...")
        
        # Drop the new table if it exists
        try:
            cursor.execute("DROP TABLE IF EXISTS messages_new")
            print("Dropped existing messages_new table")
        except Exception as e:
            print(f"Warning: Could not drop messages_new table: {str(e)}")
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages_new (
            id INT AUTO_INCREMENT PRIMARY KEY,
            chat_id VARCHAR(36) NOT NULL,
            sender VARCHAR(10) NOT NULL,
            message TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_corrected BOOLEAN DEFAULT FALSE,
            is_correction BOOLEAN DEFAULT FALSE,
            corrected_message_id INT,
            corrected_by VARCHAR(36),
            FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
        )
        """)
        
        # Copy data from the old table to the new table
        print("Copying existing messages data...")
        cursor.execute("""
        INSERT IGNORE INTO messages_new 
        (chat_id, sender, message, created_at, is_corrected, is_correction)
        SELECT chat_id, sender, message, created_at, is_corrected, is_correction
        FROM messages
        """)
        
        # Get count of copied messages
        cursor.execute("SELECT COUNT(*) FROM messages_new")
        new_count = cursor.fetchone()[0]
        print(f"Copied {new_count} messages to new table")
        
        # Check if backup table exists and drop it if it does
        cursor.execute("SHOW TABLES LIKE 'messages_backup'")
        if cursor.fetchone():
            print("Dropping existing messages_backup table")
            cursor.execute("DROP TABLE messages_backup")
        
        # Backup the old table just in case
        print("Backing up old messages table...")
        cursor.execute("RENAME TABLE messages TO messages_backup")
        
        # Rename the new table to become the main table
        print("Activating new messages table...")
        cursor.execute("RENAME TABLE messages_new TO messages")
        
        # Verify the new table
        cursor.execute("DESCRIBE messages")
        columns = cursor.fetchall()
        print("New messages table structure:")
        for col in columns:
            print(f"  {col}")
        
        conn.commit()
        print("Database fix completed successfully")
        
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error fixing database: {str(e)}")
        try:
            conn.rollback()
        except:
            pass
        return False

if __name__ == "__main__":
    print("Starting database fix script...")
    result = fix_messages_table()
    if result:
        print("Successfully fixed messages table!")
    else:
        print("Failed to fix messages table.") 