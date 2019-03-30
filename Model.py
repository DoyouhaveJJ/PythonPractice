import  tensorflow as tf
import  numpy as np
import CarData

data = CarData.load_data(download=False)
new_data = CarData.convert2onehot(data)

new_data = new_data.values.astype(np.float32)
np.random.shuffle(new_data)#打乱顺序

sep =int(0.7*len(new_data))
train_data=new_data[:sep]
test_data=new_data[sep:]

tf_input = tf.placeholder(tf.float32,[None,25],name='input')
tfx = tf_input[:,:21]
tfy = tf_input[:,21:]

l1 = tf.layers.dense(tfx,128,tf.nn.relu,name='l1')
l2 = tf.layers.dense(l1,128,tf.nn.relu,name='l2')

output = tf.layers.dense(l2,4,name='l3')

prediction = tf.nn.softmax(output,name='prediction')

loss = tf.losses.softmax_cross_entropy(onehot_labels=tfy,logits=output)

accuracy = tf.metrics.accuracy(
    labels=tf.argmax(tfy,axis=1),
    predictions = tf.argmax(output,axis=1)
)[1]

opt = tf.train.GradientDescentOptimizer(0.1)
train_op = opt.minimize(loss)

sess = tf.Session()
sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))

for t in range(4000):
    batch_index = np.random.randint(len(train_data),size=32)
    sess.run(train_op,feed_dict={tf_input:train_data[batch_index]})
    if t%50==0:
        acc_,pred_,loss_ = sess.run([accuracy,prediction,loss],
                                    feed_dict={tf_input:test_data})
        print("Step:%i" %t ,"|Accuarcy:%.2f" %acc_ ,"|Loss:%.2f" %loss_)



