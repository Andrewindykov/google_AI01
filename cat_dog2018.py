import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import model
import os


# Выберите картинку из указанного каталога
def get_one_image(train):
    files = os.listdir(train)
    n = len(files)
    ind = np.random.randint(0, n)
    img_dir = os.path.join(train, files[ind])
    image = Image.open(img_dir)
    plt.imshow(image)
    plt.show()
    image = image.resize([208, 208])
    image = np.array(image)
    return image


def evaluate_one_image():
    # Хранится путь с изображением кота и собаки, который я скачал с Baidu
    train = 'E:\\Pycharm\\tf-01\\Bigwork\\data_prepare\\test\\'
    image_array = get_one_image(train)

    with tf.Graph().as_default():
        BATCH_SIZE = 1  # Поскольку читается только одна картинка, пакетный режим установлен на 1
        N_CLASSES = 2  # 2 выходных нейрона, [1, 0] или [0, 1] вероятность кошки и собаки
        # Конверсионный формат изображения
    image = tf.cast(image_array, tf.float32)
    # Стандартизация изображения
    # image = tf.image.per_image_standardization (image) # Не добавляйте эту строку, картинки, которые мы обучили, не стандартизированы, эта строка меняла меня на два дня
    # Первоначально картинка была трехмерной [208, 208, 3]. Переопределите форму изображения для четырехмерного четырехмерного тензора.
    image = tf.reshape(image, [1, 208, 208, 3])
    logit = model.inference(image, BATCH_SIZE, N_CLASSES)
    # Поскольку при возвращении логического вывода не используется функция активации, результат активируется с помощью softmax здесь
    logit = tf.nn.softmax(logit)

    # Ввод данных в модель с наиболее примитивным заполнителем входных данных
    x = tf.placeholder(tf.float32, shape=[208, 208, 3])

    # Путь, где мы храним модель
    logs_train_dir = 'E:\\Pycharm\\tf-01\\Bigwork\\savenet02\\'
    # Определить заставку
    saver = tf.train.Saver()

    with tf.Session() as sess:
        print("Загрузить        модель        по        указанному        пути...")
        # Загрузить модель в sess
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Модель успешно загружена, шаги обучения:% s' % global_step)
    else:
    print('Ошибка загрузки модели ,, файл не найден')
    # Ввести изображение в расчет модели
    prediction = sess.run(logit, feed_dict={x: image_array})
    # Получить индекс максимальной вероятности на выходе
    max_index = np.argmax(prediction)
    print('Вероятность кошки% .6f' % прогноз[:, 0])
    print('Вероятность собаки% .6f' % прогноз[:, 1])
    # Тест
    print("I'm runing")
    evaluate_one_image()