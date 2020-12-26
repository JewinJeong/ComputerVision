import numpy as np
import pymysql
import time
import statsmodels.api as sm

def multi_load_dbscore_data():
    conn = pymysql.connect(host='localhost', user='root', password='jj8575412', db='db_score')
    curs = conn.cursor(pymysql.cursors.DictCursor)

    sql = "select * from score"
    curs.execute(sql)

    data = curs.fetchall()

    curs.close()
    conn.close()

    X = [ (t['attendance'], t['homework'], t['midterm'] ) for t in data ]
    X = np.array(X)

    y = [(t['score']) for t in data]
    y = np.array(y)

    return X, y


def multi_gradient_descent_vectorized(X, y):
    epochs = 1000000
    min_grad = 0.0001
    learning_rate = 0.001
    X = np.transpose(X)
    m = np.array([0.,0.,0.])
    m_grad = np.array([0.,0.,0.])

    c = 0.0

    n = len(y)

    c_grad = 0.0

    for epoch in range(epochs):
        y_pred = m[0]*X[0] + m[1]*X[1] + m[2]*X[2] + c
        m_grad[0] = (2*(y_pred - y) * X[0]).sum()/n
        m_grad[1] = (2*(y_pred - y) * X[1]).sum()/n
        m_grad[2] = (2*(y_pred - y) * X[2]).sum()/n
        c_grad = (2*(y_pred - y)).sum()/n

        m = m - learning_rate * m_grad
        c = c - learning_rate * c_grad

        if (epoch % 1000 == 0):
            print("epoch %d: m1_grad=%f,m2_grad=%f, m3_grad=%f, c_grad=%f, m1=%f,m2=%f, m3=%f, c=%f" % (epoch, m_grad[0],m_grad[1], m_grad[2], c_grad, m[0],m[1],m[2], c))

        if (abs(m_grad[0]) < min_grad and abs(m_grad[1]) < min_grad and abs(m_grad[2]) < min_grad and abs(c_grad) < min_grad):
            break

    return m[0], m[1], m[2] , c

X,y = multi_load_dbscore_data()
X_const = sm.add_constant(X)

model = sm.OLS(y,X_const)
ls = model.fit()

print(ls.summary())
ls_c = ls.params[0]
ls_m1 = ls.params[1]
ls_m2 = ls.params[2]
ls_m3 = ls.params[3]


X,y = multi_load_dbscore_data()
start_time = time.time()
m1, m2, m3, c = multi_gradient_descent_vectorized(X, y)
end_time = time.time()

print("%f seconds" % (end_time - start_time))

print("\n\nFinal:")
print("gdn_m1=%f, gdn_m2=%f, gdn_m3=%f, gdn_c=%f" % (m1, m2, m3, c))
print("ls_m1=%f, ls_m2=%f, ls_m3=%f, ls_c=%f" % (ls_m1, ls_m2, ls_m3, ls_c))
