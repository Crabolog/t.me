import psycopg2

conn = psycopg2.connect(database="neondb",
                        host="ep-lucky-sea-840602.eu-central-1.aws.neon.tech",
                        user="Crabolog",
                        password="EF6TAl7jwbRu",
                        port="5432")

cursor = conn.cursor()
text = '3'

cursor.execute("update zrada_level set value = 3  WHERE id = 1")
cursor.execute("SELECT * FROM zrada_level ")
print(cursor.fetchone())
conn.commit()
cursor.close()