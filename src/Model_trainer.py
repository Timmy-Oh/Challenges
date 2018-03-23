
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint
            
class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))

def schedule(ind):
    a = [0.001, 0.0008, 0.0006, 0.0004, 0.0002, 0.0001, 0.00005, 0.003, 0.0005, 0.0001, 0.00005,
         0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005]
    return a[ind]
        
def model_train_cv(model, X_tra, X_val, y_tra, y_val, x_test, model_name, batch_size = 32, epochs = 2, lr_schedule=True):
    file_path = "best_model.hdf5"
    
    RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
    check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1, save_best_only = True, mode = "min")
    lr_s = LearningRateScheduler(schedule)
    
    if lr_schedule:
        hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
                             callbacks = [RocAuc, lr_s, check_point], verbose=2)
    else:
        print('== no learing schedule')
        hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
                             callbacks = [RocAuc, check_point], verbose=2)
        
    model.load_weights(file_path)
    oof = model.predict(X_val, batch_size=batch_size, verbose=1)
    pred = model.predict(x_test, batch_size=batch_size, verbose=1)
    
    return pred, oof