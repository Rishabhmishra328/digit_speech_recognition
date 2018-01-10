import tflearn as tfl
import speech_data as sdata

learning_rate = 0.0001
iterations = 300
width = 20
height = 80
classes = 10
batch_size = 32

batch = word_batch = sdata.mfcc_batch_generator(batch_size)
X, Y = next(batch)
trainX, trainY = X, Y
testX, testY = X, Y

#Network
net = tfl.input_data([None, width, height])
net = tfl.lstm(net, 128, dropout=0.8)
net = tfl.fully_connected(net, classes, activation='softmax')
net = tfl.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')

model = tfl.DNN(net, tensorboard_verbose=0)
while 1: #training_iters
  model.fit(trainX, trainY, n_epoch=10, validation_set=(testX, testY), show_metric=True,
          batch_size=batch_size)
  _y=model.predict(X)
model.save("tflearn.lstm.model")
print (_y)
print (y)