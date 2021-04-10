import pymysql

conn=pymysql.connect(host='localhost',user='hyerim',password='hyerim',db='practice')
curs=conn.cursor(pymysql.cursors.DictCursor)
curs.execute("SELECT sno,midterm,final FROM db_score")
myresult = curs.fetchall()
a=[]
for x in range(92):
    temp=myresult[x]
    if(temp['midterm']>=20 and temp['final']>=20):
        a.append(x)
print(a)
for i in range(len(a)):
    print(myresult[a[i]])
