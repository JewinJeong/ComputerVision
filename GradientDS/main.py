import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymysql
import time
import matplotlib
matplotlib.use('Agg')
from celluloid import Camera

def simple_load_dbscore_data():
    conn = pymysql.connect(host='localhost', user='root', password='jj8575412', db='db_score')
    curs = conn.cursor(pymysql.cursors.DictCursor)

    sql = "select * from score"
    curs.execute(sql)

    data = curs.fetchall()

    curs.close()
    conn.close()

    # X = [ (t['attendance'], t['homework'], t['midterm'] ) for t in data ]
    X = [(t['midterm']) for t in data]
    X = np.array(X)

    y = [(t['score']) for t in data]
    y = np.array(y)

    return X, y

def simple_gradient_descent_vectorized(X, y):
    epochs = 100000
    min_grad = 0.0001
    learning_rate = 0.001

    m = 0.0
    c = 0.0

    n = len(y)

    c_grad = 0.0
    m_grad = 0.0

    fig, ax = plt.subplots()
    camera = Camera(fig)

    for epoch in range(epochs):

        y_pred = m * X + c
        m_grad = (2 * (y_pred - y) * X).sum() / n
        c_grad = (2 * (y_pred - y)).sum() / n

        m = m - learning_rate * m_grad
        c = c - learning_rate * c_grad

        if (epoch % 1000 == 0):
            print("epoch %d: m_grad=%f, c_grad=%f, m=%f, c=%f" % (epoch, m_grad, c_grad, m, c))

            ax.scatter(X, y)
            ax.plot(X, y_pred, color='red')
            ax.annotate("m = %.4f c = %.4f " % (m, c), xy=(0, 88), fontsize=10)
            camera.snap()

        if (abs(m_grad) < min_grad and abs(c_grad) < min_grad):
            break

    animation = camera.animate()
    animation.save("Gradient.gif", fps=10)
    return m, c

X, y = simple_load_dbscore_data()
print(X)
start_time = time.time()
m, c = simple_gradient_descent_vectorized(X, y)
end_time = time.time()

print("%f seconds" %(end_time - start_time))

print("\n\nFinal:")
print("gdv_m=%f, gdv_c=%f" %(m, c) )


