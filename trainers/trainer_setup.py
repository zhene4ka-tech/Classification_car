from tensorflow.keras.optimizers import Adam, Adadelta, SGD, Nadam
import matplotlib.pyplot as plt

optimisers= {'adam': Adam,
             'adadelta': Adadelta}

def plot_history(history):
  plt.figure()
  fig, ax = plt.subplots(figsize=(7, 5))
  plt.style.use(['dark_background'])
  plt.ylabel("Функция потерь (обучение и валидация)")
  plt.xlabel("Эпохи")
  plt.ylim([0,2])
  plt.plot(history.history["loss"])
  plt.plot(history.history["val_loss"])
  plt.figure()
  fig, ax = plt.subplots(figsize=(7, 5))
  plt.style.use(['dark_background'])
  plt.ylabel("Точность (обучение и валидация)")
  plt.xlabel("Эпохи")
  plt.ylim([0,1])
  plt.plot(history.history["accuracy"])
  plt.plot(history.history["val_accuracy"])
  plt.show()