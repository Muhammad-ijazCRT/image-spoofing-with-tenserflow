from django.shortcuts import render
# Create your views here.

from django.core.files.storage import FileSystemStorage

from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import json
from tensorflow import Graph


import tensorflow as tf
gpuoptions = tf.compat.v1.GPUOptions(allow_growth=True)
Session = tf.compat.v1.Session()
# K.set_session(session)
# classifier = Sequential()
from django.shortcuts import redirect

img_height, img_width = 224, 224
with open('./models/imagenet_classes.json', 'r') as f:
    labelInfo = f.read()

labelInfo = json.loads(labelInfo)


model_graph = Graph()
with model_graph.as_default():
    tf_session = tf.compat.v1.Session()
    with tf_session.as_default():
        model = load_model('./models/MobileNetModelImagenet.h5')


def index(request):
    context = {'a': 1}
    return render(request, 'index.html', context)

def detection(request):
    
    return render(request, 'detection.html')


def predictImage(request):
    if request.method == 'POST':
        print(request)
        print(request.POST.dict())
        
        fileObj = request.FILES['filePath']
    
        fs = FileSystemStorage()
        filePathName = fs.save(fileObj.name, fileObj)
        filePathName = fs.url(filePathName)
        testimage = '.'+filePathName
        img = image.load_img(testimage, target_size=(img_height, img_width))
        x = image.img_to_array(img)
        x = x/255
        x = x.reshape(1, img_height, img_width, 3)
        with model_graph.as_default():
            with tf_session.as_default():
                predi = model.predict(x)
                ij = predi.tolist()
                # assign values
                real_predValue = float(format(predi[0][1], ".2f")) * 100 
                fake_predValue = float(format(predi[0][0], ".2f")) * 100

                # az = predi.flat[0]
                if float(format(predi[0][0], ".2f")) > 0.7:
                    print('the images is fake...')
                    result = 'Fake'
                else:
                    print('the images is Reaal...')
                    result = 'Real'
                print('Round off Firsty value :', format(predi[0][0], ".2f"))
                print('Round of second value : ',format(predi[0][1], ".2f"))

        

        import numpy as np
        predictedLabel = labelInfo[str(np.argmax(predi[0]))]

        context = {'filePathName': filePathName,
                'predictedLabel':result, 'real_predValue':real_predValue, 'fake_predValue':fake_predValue}
        return render(request, 'result.html', context)
    else:
        return redirect('/')


def viewDataBase(request):
    import os
    listOfImages = os.listdir('./media/')
    listOfImagesPath = ['./media/'+i for i in listOfImages]
    context = {'listOfImagesPath': listOfImagesPath}
    return render(request, 'viewDB.html', context)
