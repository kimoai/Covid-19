import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix


eval_im=np.load('./split/eval_im.npy')
eval_lbl=np.load('./split/eval_lbl.npy')

eval_lbl=tf.one_hot(eval_lbl,2)

model=tf.keras.models.load_model('./model/covid.hdf5')
results=model.predict(tf.image.per_image_standardization(tf.cast(eval_im, tf.float32)))
lbl=np.argmax(eval_lbl,axis=1)
pred=np.argmax(results,axis=1)
cm=confusion_matrix(lbl,pred)
total=sum(sum(cm))

acc = (cm[0, 0] + cm[1, 1]) / total

print(cm)
print(acc)