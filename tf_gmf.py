import tensorflow as tf
import numpy as np
import heapq
import math
from time import time
from Dataset import Dataset

def get_train_instance(trainMatrix, num_negatives):
    user_input, item_input, labels = [],[],[]
    # num_users = trainMatrix.shape[0]
    for(u, i) in trainMatrix.keys():
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while trainMatrix[u,j] == 1:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

#     return loss, train_step
def getHitRatio(rankList, trueitem):
    for item in rankList:
        if item == trueitem:
            return 1
    return 0
def getNDCG(rankList, trueitem):
    for i in range(len(rankList)):
        item = rankList[i]
        if item == trueitem:
            return math.log(2)/math.log(i+2)
    return 0

def evaluate_one_rating(idx):
    rating = testRatings[idx]   #(user,item)
    items = testNegatives[idx]
    # print(rating[0])
    u = [rating[0]]*(len(items)+1)
    # print(u)
    lab = [0]*(len(items))
    lab.append(1)
    items.append(rating[1])
    pre = sess.run(predict, feed_dict={train_input_u: u, train_input_i: items, y: lab})
    map_item_score = {}
    for j in range(len(items)):
        item = items[j]
        map_item_score[item] = pre[j]
    rankList = heapq.nlargest(topk, map_item_score, key=map_item_score.get)
    hr = getHitRatio(rankList, rating[1])
    ndcg = getNDCG(rankList, rating[1])
    return (hr, ndcg)

def metrics():
    hits, ndcgs = [], []
    for i in range(num_users):
        (hr, ndcg) = evaluate_one_rating(i)
        hits.append(hr)
        ndcgs.append(ndcg)
    return (hits, ndcgs)

if __name__ == '__main__':
    sess = tf.InteractiveSession()
    dataset = Dataset("Data/ml-1m")
    t1 = time()
    trainMatrix, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = trainMatrix.shape
    print("load data done [ %fs ]. #user=%d, #item=%d, #train=%d, #test=%d" %
          (time()-t1, num_users, num_items, trainMatrix.nnz, len(testRatings)))

    num_factors = 8
    num_negatives = 4
    topk = 10
    batch_size = 256
    user_input, item_input, labels = get_train_instance(trainMatrix, num_negatives)
    #caculate the batch
    batch_index = []
    for i in range(len(user_input)):
        if i % batch_size == 0:
            batch_index.append(i)
    batch_index.append(len(user_input))
    # print(user_input[:10], item_input[:10], labels[:10])
    # loss, train_step = train()
#construct model
    train_input_u = tf.placeholder(tf.int32, shape=[None])
    train_input_i = tf.placeholder(tf.int32, shape=[None])
    y = tf.placeholder(tf.float32, shape=[None])
    yo = tf.reshape(y, [-1, 1])

    embedding_user = tf.Variable(tf.random_normal(shape=[num_users, num_factors], mean=0.0, stddev=0.01))
    embedding_item = tf.Variable(tf.random_normal(shape=[num_items, num_factors], mean=0.0, stddev=0.01))

    embed_user = tf.nn.embedding_lookup(embedding_user, train_input_u)
    embed_item = tf.nn.embedding_lookup(embedding_item, train_input_i)

    dense_input = tf.multiply(embed_user, embed_item)
    dense_w = tf.Variable(tf.random_normal(shape=[num_factors, 1], mean=0.0, stddev=0.01))
    dense_b = tf.Variable(tf.constant(0.1, shape=[1, ]))

    predict = tf.nn.sigmoid(tf.matmul(dense_input, dense_w) + dense_b)
    # loss = -tf.reduce_sum(tf.multiply(yo, tf.log(predict)) + tf.multiply(1-yo, tf.log(1-predict)))
    loss = -tf.reduce_mean(yo*tf.log(tf.clip_by_value(predict, 1e-10, 1.0)) + (1-yo)*tf.log(tf.clip_by_value(1-predict, 1e-10, 1.0)))
    # train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
    train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

    tf.global_variables_initializer().run()

    t1 = time()
    (hits, ndcgs) = metrics()
    hr, ndcg = np.mean(hits), np.mean(ndcgs)
    print("Init: HR = %.4f, NDCG = %.4f\t [%.1f s]" % (hr, ndcg, time() - t1))
    best_hr, best_ndcg, best_iter = hr, ndcg, -1

    epochs = 20
    for epoch in range(epochs):
        user_input, item_input, labels = get_train_instance(trainMatrix, num_negatives)
        l = [0]*len(batch_index)
        for step in range(len(batch_index)-1):
            t, l[step] = sess.run([train_step, loss], feed_dict={train_input_u: user_input[batch_index[step]: batch_index[step+1]], train_input_i: item_input[batch_index[step]: batch_index[step+1]], y: labels[batch_index[step]: batch_index[step+1]]})
        t2 = time()
        (hits, ndcgs) = metrics()
        hr, ndcg = np.mean(hits), np.mean(ndcgs)
        print("Iteration: %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]" % (epoch, time()-t1, hr, ndcg, l[0], time()-t2))
        if hr > best_hr:
            best_hr, best_ndcg, best_iter = hr, ndcg, epoch
    print("ENd. Best Iteration %d: HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))



