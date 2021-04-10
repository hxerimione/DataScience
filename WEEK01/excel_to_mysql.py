import pymysql
import pandas as pd


df=pd.read_excel('./db_score.xlsx')
subset = df[['sno', 'attendance', 'homework','discussion','midterm','final','score','grade']]
tuples = [tuple(x) for x in subset.values]
conn=pymysql.connect(host='localhost',user='root',password='purejang98',db='practice')
curs=conn.cursor(pymysql.cursors.DictCursor)
curs.execute("use practice")
curs.execute("drop table db_score")
curs.execute("""CREATE TABLE db_score(
     number int primary key AUTO_INCREMENT,
     sno int,
     attendance float,
     homework float,
     discussion int,
     midterm float,
     final float,
     score float,
     grade varchar(255)
     )""")

sql="INSERT INTO db_score(sno,attendance,homework,discussion,midterm,final,score,grade) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"
curs.executemany(sql,tuples)
conn.commit()
curs.execute("SELECT * FROM db_score")

myresult = curs.fetchall()

print(myresult)
conn.close()
curs.close()

