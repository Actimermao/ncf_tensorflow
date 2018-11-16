from Dataset import Dataset
from time import time
import tensorflow as tf
import scipy.sparse as sp
import numpy as np
import heapq
import math

def getHitRatio(ranklist, true_item):
    # print("example: ", ranklist, true_item)
    for i in ranklist:
        if i == true_item:
            # print("here is the true")
            # a = input("")
            return 1
    return 0
def getNDCG(ranklist, true_item):
    for i in range(len(ranklist)):
        if ranklist[i] == true_item:
            return math.log(2)/math.log(i+2)
    return 0
def evaluate_one_rating(idx):
    rating = testRatings[idx]
    true_item = rating[1]
    items = testNegatives[idx]
    users = [rating[0]]*(len(items)+1)
    labels_ = [0]*(len(items))
    labels_.append(1)
    items.append(true_item)
    pred = sess.run(predict, feed_dict={train_input_user: users, train_input_item: items, y: labels_})
    map_score = {}
    for i in range(len(items)):
        item = items[i]
        map_score[item] = pred[i]
    ranklist = heapq.nlargest(k, map_score, key=map_score.get)
    hr = getHitRatio(ranklist, true_item)
    ndcg = getNDCG(ranklist, true_item)
    return (hr, ndcg)


def metrics():
    hits, ndcgs = [], []
    for i in range(num_users):
        hit, ndcg = evaluate_one_rating(i)
        hits.append(hit)
        ndcgs.append(ndcg)
    return (hits, ndcgs)

def get_train_instance():
    user_input, item_input, labels = [], [], []
    for(u, i) in trainMatrix.keys():
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in trainMatrix:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

if __name__ == '__main__':
    dataset = Dataset("Data/ml-1m")
    t1 = time()
    trainMatrix, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = trainMatrix.shape
    print("load data done [%f s], user=%d, item=%d, train=%d, test=%d" % (time()-t1, num_users, num_items, trainMatrix.nnz, len(testRatings)))

    num_factors = 8
    num_negatives = 4
    k = 10
    batch_size = 256
    lr = 0.0006
    layers = [num_factors*8, num_factors*4, num_factors*2, num_factors]
    user_input, item_input, labels = get_train_instance()

    batch_index = []
    for i in range(len(user_input)):
        if i % batch_size == 0:
            batch_index.append(i)
    batch_index.append(len(user_input))

    train_input_user = tf.placeholder(tf.int32, shape=[None])
    train_input_item = tf.placeholder(tf.int32, shape=[None])
    y = tf.placeholder(tf.float32, shape=[None])
    yo = tf.reshape(y, [-1, 1])

    embedding_user = tf.Variable(tf.random_normal(shape=[num_users, int(layers[0]/2)], mean=0.0, stddev=0.01))
    embedding_item = tf.Variable(tf.random_normal(shape=[num_items, int(layers[0]/2)], mean=0.0, stddev=0.01))

    embed_user = tf.nn.embedding_lookup(embedding_user, train_input_user)
    embed_item = tf.nn.embedding_lookup(embedding_item, train_input_item)

    embed_concat = tf.concat([embed_user, embed_item], 1)
    w = []
    hidden = []
    for i in range(len(layers)-1):
        w.append(tf.Variable(tf.random_normal(shape=[layers[i], layers[i+1]], mean=0.0, stddev=0.01)))
        if i == 0:
            hidden.append(tf.nn.relu(tf.matmul(embed_concat, w[i])))
        else:
            hidden.append(tf.nn.relu(tf.matmul(hidden[i-1], w[i])))
    w.append(tf.Variable(tf.random_normal(shape=[layers[-1], 1], mean=0.0, stddev=0.01)))
    predict = tf.nn.sigmoid(tf.matmul(hidden[-1], w[-1]))
    loss = -tf.reduce_mean(yo*tf.log(tf.clip_by_value(predict, 1e-10, 1.0)) + (1-yo)*tf.log(tf.clip_by_value(1-predict, 1e-10, 1.0)))
    train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        epochs = 20
        hits, ndcgs = metrics()
        # print("hits_length: %d, ndcgs_length: %d"%(len(hits), len(ndcgs)))
        # print("hits: ", hits[0:10])
        hr, ndcg = np.mean(hits), np.mean(ndcgs)
        best_hr, best_ndcg, best_iteration = hr, ndcg, -1
        print("initial metrics: hr: %.4f, ndcg: %.4f " %(best_hr, best_ndcg))
        for epoch in  range(epochs):
            user_input, item_input, labels = get_train_instance()
            for step in  range(len(batch_index)-1):
                t, l = sess.run([train_step, loss], feed_dict={train_input_user: user_input[batch_index[step]: batch_index[step+1]], train_input_item: item_input[batch_index[step]: batch_index[step+1]], y:labels[batch_index[step]: batch_index[step+1]]})
            hits, ndcgs = metrics()
            hr, ndcg = np.mean(hits), np.mean(ndcgs)
            if hr > best_hr:
                best_hr, best_ndcg, best_iteration = hr, ndcg, epoch

            print("Iteration : %d, HR = %.4f NDCG = %.4f loss: %s" %(epoch, hr, ndcg, l ))
        print("END best_hr = %.4f, best_ndcg = %.4f best_iteration = %d" % (best_hr, best_ndcg, best_iteration))






