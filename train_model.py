import numpy as np
from alexnet import alexnet
WIDTH = 80
HEIGHT = 60
LR = 1e-3
EPOCHS = 6
count = 1
# MODEL_NAME = 'pygta5-{}-{}-{}-epochs-data-{}.model'.format(LR, 'alexnet',EPOCHS,count)

model = alexnet(WIDTH, HEIGHT, LR)

train_data = np.load(MODEL_NAME)

train = train_data[:int(len(train_data)*0.9)]  #90% top data
test = train_data[int(len(train_data)*0.9):]  #remaining 10% data

X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)


model.save(MODEL_NAME)
count += 1



#tensorboard --logdir=foo:F:/python_GTA/log
                    
