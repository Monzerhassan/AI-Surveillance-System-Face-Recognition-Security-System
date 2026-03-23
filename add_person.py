from database import *

conn = connect_db()
cursor = conn.cursor()

cursor.execute("INSERT OR REPLACE INTO persons VALUES (?, ?, ?, ?)",
               ("monzer", 25, "سارق", "الاتصال بالشرطة"))

conn.commit()
conn.close()

print("تمت اضافة البيانات")