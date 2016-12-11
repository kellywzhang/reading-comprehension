import tensorflow as tf 

new_saver = tf.train.import_meta_graph('/Users/varundn/CDS/Fall2016/Cho/qa/reading-comprehension/runs/1480726425/checkpoints/model-2.meta')

new_saver.restore(tf.Session(), tf.train.latest_checkpoint('/Users/varundn/CDS/Fall2016/Cho/qa/reading-comprehension/runs/1480726425/checkpoints'))