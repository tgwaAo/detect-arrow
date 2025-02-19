#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
import pathlib as pl


# In[9]:


INV_TARGET_SIZE = (68, 24)


# ## Train CNN

# ### Preview data

# In[3]:


POS_PATH = pl.Path('../dataset/arrows/')
if not POS_PATH.is_dir():
    raise IOError('path not valid')

NBR_POS_FILES = len(list(POS_PATH.iterdir()))
print(NBR_POS_FILES)


# In[4]:


NEG_PATH = pl.Path('../dataset/anything/')
if not NEG_PATH.is_dir():
    raise IOError('path not valid')

NBR_NEG_FILES = len(list(NEG_PATH.iterdir()))
print(NBR_NEG_FILES)


# In[5]:


first_file = list(POS_PATH.iterdir())[0]
str(first_file)


# In[6]:


img = cv2.imread(str(first_file), cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')


# In[7]:


img.shape


# In[8]:


get_ipython().run_line_magic('reset', '')


# ### Prepare data

# In[1]:


INV_TARGET_SIZE = (68, 24)


# In[2]:


from keras.preprocessing import image_dataset_from_directory


# In[3]:


train_ds = image_dataset_from_directory(
    '../dataset',
    label_mode='binary',
    color_mode='grayscale',
    batch_size=1_000,
    image_size=INV_TARGET_SIZE,
    seed=42,
    validation_split=0.2,
    subset='training',
)
print(train_ds.class_names)


# In[4]:


val_ds = image_dataset_from_directory(
    '../dataset',
    label_mode='binary',
    color_mode='grayscale',
    batch_size=1_000,
    image_size=INV_TARGET_SIZE,
    seed=42,
    shuffle=True,
    validation_split=0.2,
    subset='validation',
)
print(val_ds.class_names)


# In[5]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"), cmap='gray')
        plt.title(train_ds.class_names[int(labels[i])])
        plt.axis("off")
    break


# ### Prepare model

# In[14]:


from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Rescaling, Input
import numpy as np
import matplotlib.pyplot as plt


# In[15]:


EPOCHS = 10


# In[16]:


model = Sequential([
    Input(shape=(68, 24, 1)),
    Rescaling(1./255),
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='same'),
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='same'),

    Flatten(),

    Dense(100, activation='relu'),
    Dropout(0.05),
    Dense(1, activation='sigmoid'),
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())


# In[17]:


history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=EPOCHS
)
model.save('../model/arrow_detection.keras')

plt_epochs = range(1, (EPOCHS + 1))
loss_values = history.history['loss']
val_loss_values = history.history['val_loss']

line1 = plt.plot(plt_epochs, val_loss_values, label='Validation/Test Loss')
line2 = plt.plot(plt_epochs, loss_values, label='Training Loss')
plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)
plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)
plt.xlabel('Epochs') 
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()

acc_values = history.history['accuracy']
val_acc_values = history.history['val_accuracy']

line1 = plt.plot(plt_epochs, val_acc_values, label='Validation/Test Accuracy')
line2 = plt.plot(plt_epochs, acc_values, label='Training Accuracy')
plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)
plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)
plt.xlabel('Epochs') 
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.show()


# In[1]:


from keras.models import load_model
model = load_model('../model/arrow_detection.keras')


# In[9]:


for images, labels in val_ds.take(1):
    test_image = images[0].numpy()
    test_label = labels[0].numpy()
    break
    
prediction = model.predict(test_image[None])
print(f'prediction:{prediction[0][0]:.2f}')
print(f'real:{test_label}')
plt.imshow(test_image, cmap='gray')
plt.show()


# In[10]:


from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np


# In[11]:


res_imgs = []
res_labels = []

for images, labels in val_ds:
    res_imgs.extend(images)
    res_labels.extend(labels)

images = np.array(res_imgs)
labels = np.concatenate(res_labels, axis=0)
print(images.shape)
print(labels.shape)


# In[12]:


set(labels)


# In[13]:


predictions = (model.predict(images) > 0.5).astype("int32")
print(classification_report(labels, predictions))


# ## Saliency

# In[6]:


from keras.models import load_model
model = load_model('../model/arrow_detection.keras')


# In[10]:


res_imgs = []
res_labels = []

for images, labels in val_ds:
    res_imgs.extend(images)
    res_labels.extend(labels)

images = np.array(res_imgs)
labels = np.concatenate(res_labels, axis=0)
print(images.shape)
print(labels.shape)


# In[11]:


set(labels)


# In[15]:


print(images.shape)
saliency_part = [0] * 10
counter = 0

for num in range(len(labels)):
    if labels[num] == 1:
        saliency_part[counter] = images[num]
        counter += 1
        if counter == 5:
            break

for num in range(len(labels)):
    if labels[num] == 0:
        saliency_part[counter] = images[num]
        counter += 1
        if counter == 10:
            break

print(counter)
saliency_part = np.asarray(saliency_part)
for i in saliency_part:
    plt.imshow(i, cmap='gray')
    plt.show()


# In[16]:


try:
    import tensorflow.keras
except ImportError:
    from tensorflow import keras
    import tensorflow
    
    import sys
    
    tensorflow.keras = keras
    tensorflow.keras.backend = keras.backend
    
    sys.modules['tensorflow.keras'] = sys.modules['keras']
    sys.modules['tensorflow.keras.backend'] = 'keras hack'

if 'output_names' not in dir(model):
    model.output_names = [layer.name for layer in model.layers]


# In[17]:


from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

replace2linear = ReplaceToLinear()


# In[18]:


def score_function(output):
    return (output[0][0], output[1][0], output[2][0], output[3][0], output[3][0], output[4][0], output[5][0], output[6][0], output[7][0], output[8][0], output[9][0])


# In[19]:


from tensorflow.keras import backend as K
from tf_keras_vis.saliency import Saliency

saliency = Saliency(model,
                    model_modifier=replace2linear,
                    clone=False)


# In[20]:


saliency_map = saliency(score_function, saliency_part)
saliency_map.shape


# In[21]:


# Render
image_titles = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
f, ax = plt.subplots(nrows=2, ncols=len(image_titles), figsize=(12, 4))
for i, title in enumerate(image_titles):
    ax[0, i].set_title(title, fontsize=16)
    ax[0, i].imshow(saliency_map[i], cmap='jet')
    ax[0, i].axis('off')
    ax[1, i].imshow(saliency_part[i], cmap='gray')
    ax[1, i].axis('off')
plt.tight_layout()
plt.show()


# ### Test single image

# In[22]:


get_ipython().run_line_magic('reset', '')


# In[1]:


img_filename = '../example-images/50_cm.jpg'


# In[3]:


img_filename = '../example-images/multiple.jpg'


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import cv2

if not 'model' in dir():
    print('load model...')
    from keras.models import load_model
    model = load_model('../model/arrow_detection.keras')
    print('loaded model')
    
else:
    print('model already loaded')
    
model.trainable = False

COMPARED_SIZE = (24, 68)
AREA_BORDER = COMPARED_SIZE[0] * COMPARED_SIZE[1]
WIDTH_TO_HEIGHT = COMPARED_SIZE[0] / COMPARED_SIZE[1]
SIZE_FACTOR = 0.3
MIN_WIDTH_TO_HEIGHT = WIDTH_TO_HEIGHT * (1 - SIZE_FACTOR)
MAX_WIDTH_TO_HEIGHT = WIDTH_TO_HEIGHT * (1 + SIZE_FACTOR)

def prepare_rotation(min_rect):
    """
    Prepare portrait rotation. No difference between up and down.
    """
    
    width_to_height = min_rect[1][0] / min_rect[1][1]
    
    if width_to_height >= 1:
        return 90 - min_rect[2]
    else:
        return min_rect[2]

def rotate_and_crop_min_rect(image, min_area_rect):
    factor = 1.3

    box = cv2.boxPoints(min_area_rect)
    box = np.intp(box)

    width = round(min_area_rect[1][0])
    height = round(min_area_rect[1][1])

    size_of_transformed_image = max(min_area_rect[1])
    min_needed_height = int(np.sqrt(2 * np.power(size_of_transformed_image, 2)))

    #angle = prepare_rotation(min_area_rect)
    width_to_height = min_area_rect[1][0] / min_area_rect[1][1]
    
    if width_to_height >= 1:
        angle = -1 * (90 - min_rect[2])
    else:
        angle = min_rect[2]    
        
    size = (min_needed_height, min_needed_height)

    x_coordinates_of_box = box[:,0]
    y_coordinates_of_box = box[:,1]
    x_min = min(x_coordinates_of_box)
    x_max = max(x_coordinates_of_box)
    y_min = min(y_coordinates_of_box)
    y_max = max(y_coordinates_of_box)

    center = (int((x_min+x_max)/2), int((y_min+y_max)/2))
    cropped = cv2.getRectSubPix(image, size, center) 
    M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)
    cropped = cv2.warpAffine(cropped, M, size)
    
    if width_to_height >= 1:
        cropped_rotated = cv2.getRectSubPix(cropped, (int(factor * height), int(factor * width)), (size[0]/2, size[1]/2))
    else:
        cropped_rotated = cv2.getRectSubPix(cropped, (int(factor * width), int(factor * height)), (size[0]/2, size[1]/2))

    return cropped_rotated


img = cv2.imread(img_filename)

if img is None:
    raise IOError('file not valid')

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.blur(gray_img, (3,3))

sigma = 0.33
v = np.median(blurred)

#---- apply automatic Canny edge detection using the computed median----
lower = int(max(0, (1.0 - sigma) * v))    #---- lower threshold
upper = int(min(255, (1.0 + sigma) * v))  #---- upper threshold
thresh_img = cv2.Canny(blurred, lower, upper)
cnts, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

filtered_list = []
pos_filtered_to_pos_source = {}
pos_filtered = 0
center_list = []
too_close = False
for pos_source, con in enumerate(cnts):
    min_rect = cv2.minAreaRect(con)
    center, size, angle = min_rect
    area = size[0] * size[1]

    if area < AREA_BORDER:
        continue

    low_value = min(size[0], size[1])
    high_value = max(size[0], size[1])
    width_to_height = low_value / high_value

    if MIN_WIDTH_TO_HEIGHT < width_to_height < MAX_WIDTH_TO_HEIGHT:
        for c_point in center_list:
            too_close = np.all(np.isclose(center, c_point, rtol=0, atol=20))
            if too_close:
                break

        if too_close:
            continue
        center_list.append(center)
        cropped_img = rotate_and_crop_min_rect(gray_img, min_rect)
        small_img = cv2.resize(cropped_img, COMPARED_SIZE)
        filtered_list.append(small_img)
        pos_filtered_to_pos_source[pos_filtered] = pos_source
        pos_filtered += 1

filtered_list = np.array(filtered_list)
prediction = model.predict(filtered_list)
print(prediction.shape)
print(prediction)

positive_contours = []
negative_contours = []

for pos, value in enumerate(prediction):
    idx = pos_filtered_to_pos_source[pos]
    if value[0] >= 0.5:
        positive_contours.append(cnts[idx])
    else:
        negative_contours.append(cnts[idx])

cv2.drawContours(img, positive_contours, -1, (0,0,255), 2)
cv2.drawContours(img, negative_contours, -1, (255,0,0), 2)

plt.imshow(img, cmap='gray')
plt.show()
print('done')


# ### Tests with camera

# In[ ]:


cam_nbr = 2


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
import threading
from queue import Queue

if not 'model' in dir():
    print('load model...')
    from keras.models import load_model
    model = load_model('../model/arrow_detection.keras')
    print('loaded model')

else:
    print('model already loaded')
    
model.trainable = False

COMPARED_SIZE = (24, 68)
AREA_BORDER = COMPARED_SIZE[0] * COMPARED_SIZE[1]
WIDTH_TO_HEIGHT = COMPARED_SIZE[0] / COMPARED_SIZE[1]
SIZE_FACTOR = 0.3
MIN_WIDTH_TO_HEIGHT = WIDTH_TO_HEIGHT * (1 - SIZE_FACTOR)
MAX_WIDTH_TO_HEIGHT = WIDTH_TO_HEIGHT * (1 + SIZE_FACTOR)

WINDOW_TITLE = 'detect-arrow'

def prepare_rotation(min_rect):
    """
    Prepare portrait rotation. No difference between up and down.
    """
    
    width_to_height = min_rect[1][0] / min_rect[1][1]
    
    if width_to_height >= 1:
        return 90 - min_rect[2]
    else:
        return min_rect[2]

def rotate_and_crop_min_rect(image, min_area_rect):
    factor = 1.3

    box = cv2.boxPoints(min_area_rect)
    box = np.intp(box)

    width = round(min_area_rect[1][0])
    height = round(min_area_rect[1][1])

    size_of_transformed_image = max(min_area_rect[1])
    min_needed_height = int(np.sqrt(2 * np.power(size_of_transformed_image, 2)))

    width_to_height = min_area_rect[1][0] / min_area_rect[1][1]
    
    if width_to_height >= 1:
        angle = -1 * (90 - min_rect[2])
    else:
        angle = min_rect[2]    
        
    size = (min_needed_height, min_needed_height)

    x_coordinates_of_box = box[:,0]
    y_coordinates_of_box = box[:,1]
    x_min = min(x_coordinates_of_box)
    x_max = max(x_coordinates_of_box)
    y_min = min(y_coordinates_of_box)
    y_max = max(y_coordinates_of_box)

    rotated = False
    center = (int((x_min+x_max)/2), int((y_min+y_max)/2))
  
    cropped = cv2.getRectSubPix(image, size, center) 
  
    M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)
    cropped = cv2.warpAffine(cropped, M, size)
    
    if width_to_height >= 1:
        cropped_rotated = cv2.getRectSubPix(cropped, (int(factor * height), int(factor * width)), (size[0]/2, size[1]/2))
    else:
        cropped_rotated = cv2.getRectSubPix(cropped, (int(factor * width), int(factor * height)), (size[0]/2, size[1]/2))

    return cropped_rotated


# bufferless VideoCapture
class VideoCapture:
    def __init__(self, *args):
        self.image = None
        self.stopped = False
        self.Q = Queue(maxsize=2)
        self.cap = cv2.VideoCapture(*args)
        self.lock = threading.Lock()
        self.event = threading.Event()
        self.t = threading.Thread(target=self._reader)
        self.t.start()

    def _reader(self):
        while not self.event.is_set():
            with self.lock:
                if self.Q.full():
                    _ = self.Q.get()  # remove value for new ones
                    
                ret, image = self.cap.read()
                if not ret:
                    self.cap.release()
                    raise ValueError('could not get image from VideoCapture')
                    
                self.Q.put(image)

    def read(self):
        return True, self.Q.get()

    def isOpened(self):
        return self.cap.isOpened()

    def set(self, *args):
        with self.lock:
            self.cap.set(*args)

    def release(self):
        self.event.set()
        self.t.join()
        self.cap.release()


cap = VideoCapture(cam_nbr)

if not cap.isOpened():
    cap.release()
    print('could not open camera at 2')
    raise

abort = False

while not abort:
    ret, img = cap.read()
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.blur(gray_img, (3,3))
    

    sigma = 0.33
    v = np.median(gray_img)

    #---- apply automatic Canny edge detection using the computed median----
    lower = int(max(0, (1.0 - sigma) * v))    #---- lower threshold
    upper = int(min(255, (1.0 + sigma) * v))  #---- upper threshold
    thresh_img = cv2.Canny(gray_img, lower, upper)
    cnts, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    filtered_list = []
    pos_filtered_to_pos_source = {}
    pos_filtered = 0
    center_list = []
    too_close = False
    for idx, con in enumerate(cnts):
        min_rect = cv2.minAreaRect(con)
        center, size, angle = min_rect
        area = size[0] * size[1]
    
        if area < AREA_BORDER:
            continue
    
        low_value = min(size[0], size[1])
        high_value = max(size[0], size[1])
        width_to_height = low_value / high_value

        # start prediction
        if MIN_WIDTH_TO_HEIGHT < width_to_height < MAX_WIDTH_TO_HEIGHT:
            for c_point in center_list:
                too_close = np.all(np.isclose(center, c_point, rtol=0, atol=20))
                if too_close:
                    break

            if too_close:
                continue
            center_list.append(center)
            cropped_img = rotate_and_crop_min_rect(gray_img, min_rect)
            small_img = cv2.resize(cropped_img, COMPARED_SIZE)
            filtered_list.append(small_img)
            pos_filtered_to_pos_source[pos_filtered] = idx
            pos_filtered += 1

    if filtered_list:
        filtered_list = np.array(filtered_list)            
        prediction = model.predict(filtered_list, verbose=0)
        
        positive_contours = []
        negative_contours = []
        
        for pos, value in enumerate(prediction):
            idx = pos_filtered_to_pos_source[pos]
            if value[0] >= 0.5:
                positive_contours.append(cnts[idx])
            else:
                negative_contours.append(cnts[idx])
    else:
        negative_contours = cnts    
        
    cv2.drawContours(img, positive_contours, -1, (255,0,0), 2)
    cv2.drawContours(img, negative_contours, -1, (0,0,255), 2)
                    
    cv2.imshow(WINDOW_TITLE, img)
    key = cv2.waitKey(25) & 0xFF

    if key == 27 or key == 113:
        print('aborting')
        abort = True
        break

cap.release()
cv2.destroyAllWindows()
print('done')


# ## Commentary

# The f1-score is not realistic, which is shown in a camera test.  
# Further optimization should be done.  
# maybe later tfma  
