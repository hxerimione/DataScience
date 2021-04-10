import pymysql
import pandas as pd


df=pd.read_excel('./db_score_3_labels.xlsx')
subset = df[['sno','homework','discussion','midterm','grade']]
tuples = [tuple(x) for x in subset.values]
conn=pymysql.connect(host='localhost',user='root',password='purejang98',db='practice')
curs=conn.cursor(pymysql.cursors.DictCursor)
curs.execute("use practice")
curs.execute("drop table ds_exam")
curs.execute("""CREATE TABLE ds_exam(
     number int primary key AUTO_INCREMENT,
     sno int,
     homework float,
     discussion int,
     midterm float,
     grade varchar(255)
     )""")

sql="INSERT INTO ds_exam(sno,homework,discussion,midterm,grade) VALUES (%s,%s,%s,%s,%s)"
curs.executemany(sql,tuples)
conn.commit()
curs.execute("SELECT * FROM ds_exam")

myresult = curs.fetchall()

print(myresult)
conn.close()
curs.close()

print(type(myresult))
