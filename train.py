import numpy as np
import tensorflow as tf
from model import *

from utils import *

batch_size = 50


def train():
    train_img_paths,train_atractive_label,train_male_label,train_smiling_label,train_young_label,\
           test_img_paths,test_atractive_label,test_male_label,test_smiling_label,test_young_label=load_data_label()

    with tf.name_scope('place_holder'):
        x = tf.placeholder(tf.float32,shape=[None,160,160,3])
        atractive_y = tf.placeholder(tf.float32,shape=[None,2])
        male_y = tf.placeholder(tf.float32,shape=[None,2])
        smiling_y = tf.placeholder(tf.float32,shape=[None,2])
        young_y = tf.placeholder(tf.float32,shape=[None,2])
        ph = tf.placeholder(tf.float32)

    atractive_out,male_out,smiling_out,young_out = inference2(x, ph, phase_train=True,
              bottleneck_layer_size=1024, weight_decay=0.0, reuse=None)
    optimizer = tf.train.AdamOptimizer(1e-3)


    with tf.name_scope('atractive_loss'):
        loss_atractive = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=atractive_y,logits=atractive_out))
        correct_prediction = tf.equal(tf.argmax(atractive_out, 1), tf.argmax(atractive_y, 1))
        accuracy_atractive = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    trainop_atractive = optimizer.minimize(loss_atractive)

    with tf.name_scope('male_loss'):
        loss_male = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=male_y,logits=male_out))
        correct_prediction = tf.equal(tf.argmax(male_out, 1), tf.argmax(male_y, 1))
        accuracy_male = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    trainop_male = optimizer.minimize(loss_male)

    with tf.name_scope('smiling_loss'):
        loss_smiling = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=smiling_y,logits=smiling_out))
        correct_prediction = tf.equal(tf.argmax(smiling_out, 1), tf.argmax(smiling_y, 1))
        accuracy_smiling = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    trainop_smiling = optimizer.minimize(loss_smiling)

    with tf.name_scope('young_loss'):
        loss_young = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=young_y,logits=young_out))
        correct_prediction = tf.equal(tf.argmax(young_out, 1), tf.argmax(young_y, 1))
        accuracy_young = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    trainop_young = optimizer.minimize(loss_young)

    ep = 0
    ll = len(train_img_paths)
    ll_t = len(test_img_paths)
    test_batch_num = int(ll_t/batch_size)

    train_img_paths,train_atractive_label,train_male_label,train_smiling_label,train_young_label = shuffle(train_img_paths,train_atractive_label,train_male_label,train_smiling_label,train_young_label)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print('-'*100)
        print('-'*100)
        print('epoch: ',ep)
        print('-'*100)
        print('-'*100)

        for step in range(50000):
            start = (step*batch_size)%ll
            end = start+batch_size

            if end>=ll:
                print('-'*100)
                print('-'*100)
                ep = ep + 1
                print('epoch: ',ep)
                print('-'*100)
                print('-'*100)
                start = 0
                end = start+batch_size

                train_img_paths,train_atractive_label,train_male_label,train_smiling_label,train_young_label = shuffle(train_img_paths,train_atractive_label,train_male_label,train_smiling_label,train_young_label)


            batch_data,batch_label_attrative,batch_label_male,batch_label_smiling,batch_label_young = get_batch(train_img_paths,train_atractive_label,train_male_label,train_smiling_label,train_young_label,start,end)
            feed_dict = {x:batch_data,atractive_y:batch_label_attrative,male_y:batch_label_male,smiling_y:batch_label_smiling,young_y:batch_label_young,ph:0.7}

            _ = sess.run(trainop_atractive,feed_dict= feed_dict)
            _ = sess.run(trainop_male,feed_dict= feed_dict)
            _ = sess.run(trainop_smiling,feed_dict= feed_dict)
            _ = sess.run(trainop_young,feed_dict= feed_dict)


            if (step+1) % 30 == 0:
                feed_dict = {x:batch_data,atractive_y:batch_label_attrative,male_y:batch_label_male,smiling_y:batch_label_smiling,young_y:batch_label_young,ph:1.0}

                acc_atractive=sess.run(accuracy_atractive,feed_dict= feed_dict)
                acc_male=sess.run(accuracy_male,feed_dict= feed_dict)
                acc_smiling=sess.run(accuracy_smiling,feed_dict= feed_dict)
                acc_young=sess.run(accuracy_young,feed_dict= feed_dict)

                print('step: ',step, 'acc_atractive:',  acc_atractive)
                print('step: ',step, 'acc_male:' ,  acc_male)
                print('step: ',step, 'acc_smiling:' , acc_smiling)
                print('step: ',step, 'acc_young:' , acc_young)

                print('-'*100)
                print('-'*100)

                test_acc_atractive = 0
                test_acc_male = 0
                test_acc_smiling = 0
                test_acc_young = 0

                for ii in range(test_batch_num):
                    start = ii*batch_size
                    end = start+batch_size
                    batch_data,batch_label_attrative,batch_label_male,batch_label_smiling,batch_label_young = get_batch(test_img_paths,test_atractive_label,test_male_label,test_smiling_label,test_young_label,start,end)
                    feed_dict = {x:batch_data,atractive_y:batch_label_attrative,male_y:batch_label_male,smiling_y:batch_label_smiling,young_y:batch_label_young,ph:1}

                    acc_atractive=sess.run(accuracy_atractive,feed_dict= feed_dict)
                    acc_male=sess.run(accuracy_male,feed_dict= feed_dict)
                    acc_smiling=sess.run(accuracy_smiling,feed_dict= feed_dict)
                    acc_young=sess.run(accuracy_young,feed_dict= feed_dict)

                    test_acc_atractive += acc_atractive
                    test_acc_male += acc_male
                    test_acc_smiling += acc_smiling
                    test_acc_young += acc_young

                test_acc_atractive /= test_batch_num
                test_acc_male /= test_batch_num
                test_acc_smiling /= test_batch_num
                test_acc_young/= test_batch_num

                print( 'test_acc_atractive:',  test_acc_atractive)
                print( 'test_acc_male:' ,  test_acc_male)
                print( 'test_acc_smiling:' , test_acc_smiling)
                print( 'test_acc_young:' , test_acc_young)


if __name__ == '__main__':
    train()
    print('finish!!')
