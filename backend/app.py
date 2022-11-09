from flask import Flask,request, url_for, redirect, render_template,jsonify
import pickle
import numpy as np
# import required packages
import argparse
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import cv2
from flask import Flask,request, url_for, redirect, render_template,jsonify
import pickle
import numpy as np
# import required packages
import cv2
import argparse
import numpy as np
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS
from PIL import Image
import pytesseract
from pytesseract import Output
import re
from io import BytesIO
import glob
import requests
import pandas as pd
import cv2
from pytesseract import Output
import re
from os.path import join, dirname, realpath
import base64
import io  # from native modules
from PIL import Image, ImageFile  # from Pillow
from colorama import Fore  # from native modules
import platform  # from native modules
from textblob import TextBlob
import PIL
import glob
import requests
import pandas as pd
import re
from os.path import join, dirname, realpath
import base64
import json
from detect import start
import io  # from native modules
from PIL import Image, ImageFile  # from Pillow
from colorama import Fore  # from native modules
import platform  # from native modules
import PIL
import nltk
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
import re
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
lm= WordNetLemmatizer()
nltk.download('wordnet')
stopwords.words("english")

ImageFile.LOAD_TRUNCATED_IMAGES = True
#tesseract-ocr local path
# pytesseract.pytesseract.tesseract_cmd = '/app/.apt/usr/bin/tesseract'
#pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)

CORS(app)

classes = ['Appliances', 'Automotive', 'Electronics', 'Health Care', 'beauty',
           'food', 'industry']
pickled_model = pickle.load(open('textmodel2.pkl', 'rb'))

def categorize(final_text):
    conf = pickled_model.predict_proba([final_text])
    confidence_classes = {classes[i]: conf[0][i] for i in range(len(conf[0]))}
    print(confidence_classes)
    max_conf = max(conf[0])
    max_class = sorted(confidence_classes, key=lambda x: confidence_classes[x])[-1]
    if max(conf[0]) < 0.5:
        sorted(confidence_classes, key=lambda x: confidence_classes[x])[-1]
        return "miscellaneous",100
    else:
        return str(pickled_model.predict([final_text])[0]),max(conf[0]) * 100



# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


# skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)

    else:
        angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated


# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)




def correction(text):
    text = "".join(text)
    text = TextBlob(text).correct()
    return str(text)


def total(img):
    # Adding custom options
    custom_config = r'--oem 3 --psm 11 -c tessedit_char_blacklist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" -c tessedit_char_whitelist="0123456789$£€¥,."'
    canny = get_grayscale(img)
    h, w = canny.shape
    d = pytesseract.image_to_data(canny, config=custom_config, output_type=Output.DICT)
    d = correction(d['text'])
    d = re.findall(r'[\d,.$£€¥₹]*', d)
    d = "".join(d)
    return d

def address(img):
    # Adding custom options
    custom_config = r'--psm 11 --oem 3'
    canny = get_grayscale(img)
    ignore_words = ['your company name','from','to','ship','shipping','bill of','address','billing','bill to','ship to','shipping to','billing to']
    d = pytesseract.image_to_string(canny, config=custom_config,output_type=Output.DICT)
    print(d['text'])
    d = d['text']
    d = [x for x in d if x.lower() not in ignore_words]
    d = "".join(d)
    return d

def invoice_number(img):
    custom_config = r'--psm 11 --oem 3 -c tessedit_char_blacklist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" -c tessedit_char_whitelist="0123456789$£€¥#"'
    canny = get_grayscale(img)
    d = pytesseract.image_to_data(canny, config=custom_config, output_type=Output.DICT)
    d = correction(d['text'])
    d = re.findall(r'\d+', d)
    d = "".join(d)
    return d

def date_extract(img):
    custom_config = r'--psm 11 --oem 3'
    canny = get_grayscale(img)
    d = pytesseract.image_to_data(canny, config=custom_config, output_type=Output.DICT)
    print("".join(d['text']))
    return str("".join(d['text']))

def table(img):
    img = cv2.resize(img, (int(img.shape[1] + (img.shape[1] * .1)),
                           int(img.shape[0] + (img.shape[0] * .25))),
                     interpolation=cv2.INTER_AREA)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    custom_config = r'-l eng --oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-:.$%.,/@& *"'
    d = pytesseract.image_to_data(img_rgb, config=custom_config, output_type=Output.DICT)
    df = pd.DataFrame(d)

    df1 = df[(df.conf != '-1') & (df.text != ' ') & (df.text != '')]
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    sorted_blocks = df1.groupby('block_num').first().sort_values('top').index.tolist()
    for block in sorted_blocks:
        curr = df1[df1['block_num'] == block]
        sel = curr[curr.text.str.len() > 3]
        # sel = curr
        char_w = (sel.width / sel.text.str.len()).mean()
        prev_par, prev_line, prev_left = 0, 0, 0
        text = ''
        l = []
        for ix, ln in curr.iterrows():
            if prev_par != ln['par_num']:
                text += '\n'
                prev_par = ln['par_num']
                prev_line = ln['line_num']
                prev_left = 0
            elif prev_line != ln['line_num']:
                text += '\n'
                prev_line = ln['line_num']
                prev_left = 0
            added = 0  # num of spaces that should be added
            if ln['left'] / char_w > prev_left + 1:
                added = int((ln['left']) / char_w) - prev_left
                text += ' ' * added
            text += ln['text'] + ' '

            prev_left += len(ln['text']) + added + 1

        text += '\n'
        text = text.split('\n')
        res = [x for x in text if x.strip()]
        z = res[0].split('   ')
        headers = [x for x in z if x.strip()]
        values = []
        for i in range(1, len(res)):
            values.append([x for x in res[i].split('    ') if x.strip()])
    return (headers, values)




@app.route('/')
def root():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
        print("hi")
        body = request.files
        image = body.to_dict()['files[]']
        # choice = ['invono','date','total','table','address']
        exist_classes = []
        filename = secure_filename(image.filename)
        #print(os.path.join(UPLOAD_FOLDER, filename))
        #image.save(os.path.join('/tmp',filename))

        classess = 'classes.txt'
        classes = ['address','invono','table','total','date']
        # read input image
        image2 = np.asarray(Image.open(image).convert('RGB'))
        values, img = start('invoice500.pt', image2)
        #cv2.imwrite('hi.jpg',img)

        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
        dw = image2.shape[1]
        dh = image2.shape[0]
        invonoval = ''
        addressval = ''
        dateval = ''
        totalval = ''
        tableval = None
        address_conf = 0
        invono_conf = 0
        date_conf = 0
        total_conf = 0
        table_conf = 0
        headers_table = []
        values_table = []
        for i in values:
            class_id = int(i[0])
            w = i[3]
            h = i[4]
            x = i[1]
            y = i[2]
            confidence = i[5]
            label = str(classes[class_id])

            x_center, y_center, w, h = float(x), float(y), float(w), float(h)
            x_center = round(x_center * dw)
            y_center = round(y_center * dh)
            w = round(w * dw)
            h = round(h * dh)
            x = round(x_center - w / 2)
            y = round(y_center - h / 2)
            cropped_image = image2[y:y + h, x:x + w]
            # color = COLORS[int(class_id)]
            # print("+",x+w, y+h)
            # cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            #cv2.imwrite(os.path.join(app.config['CROPPED_FOLDER'], str(classes[class_id]) + '.png'), cropped_image)
            #cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            if label == 'address':
                # s = address(cropped_image)
                address_conf = confidence
           
                addressval = address(cropped_image)
            elif label == 'total':
                
                total_conf = confidence

               

                totalval = total(cropped_image)

            elif label == 'invono':
                


                invono_conf = confidence

             

                invonoval = invoice_number(cropped_image)

            elif label == 'date':
                


                date_conf = confidence

              

                dateval = date_extract(cropped_image)

            elif label == 'table':
         
                address_conf = confidence
                tableval = table(cropped_image)
                print(tableval)
                headers_table = tableval[0]
                values_table = tableval[1]
                g = []
                for i in values_table:
                    f = ''.join([j for j in " ".join(i) if not j.isdigit()])
                    g.append(f)
                sentence = "".join(g)

                sentence = sentence.lower()
                words = nltk.word_tokenize(sentence.lower())

                new_words = [word for word in words if word.isalnum()]

                WordSet = []
                for word in new_words:
                    if word not in set(stopwords.words("english")):
                        WordSet.append(word)
                final_text = " ".join(WordSet)

                

                response_data = categorize(final_text)

                category_data = response_data[0]
                category_conf = response_data[1]



        _, im_arr = cv2.imencode('.jpg', img)  # im_arr: image in Numpy one-dim array format.
        im_bytes = im_arr.tobytes()
        im_b64 = base64.b64encode(im_bytes)
        jpg_buffer = str(im_b64)[2:len(str(im_b64)) - 1]
        # byte_io.close()
        # cv2.imwrite(os.path.join(app.config['PREDICTION_FOLDER'], name),image)
        #
        # # release resources
        # cv2.destroyAllWindows()
        return jsonify({"category":[category_data,category_conf],'date':[dateval,date_conf],'address':[addressval,address_conf],'invono':[invonoval,invono_conf],'total':[totalval,total_conf],'image':str(jpg_buffer),'headers':headers_table,'values':values_table})


@app.route('/crop',methods=['POST'])
def crop():
    body = request.files
    body2 = request.form
    image = body.to_dict()['crop[]']
    label = body2.to_dict()['label']
    cropped_image = np.asarray(Image.open(image).convert('RGB'))
    text = ''
    if label == 'address':
        # print(cropped_image.shape)
        # s = requests.post(url='https://ocr-tesseract-api2.herokuapp.com/predictocr',
        #                   json={"text_nlp[]": cropped_image.tolist(), "label": label})
        s = address(cropped_image)


        text = s
        return text
    elif label == 'total':
        s = total(cropped_image)
        # s = requests.post(url='https://ocr-tesseract-api2.herokuapp.com/predictocr',
        #                   json={"text_nlp[]": cropped_image.tolist(), "label": label})
        # s = address(cropped_image)



        text = s
        return text
    elif label == 'invono':
        # s = invoice_number(cropped_image)
        s = requests.post(url='https://ocr-tesseract-api2.herokuapp.com/predictocr',
                          json={"text_nlp[]": cropped_image.tolist(), "label": label})
        # s = address(cropped_image)


        text = s
        return text
    elif label == 'date':
        # s = date_extract(cropped_image)
        s = requests.post(url='https://ocr-tesseract-api2.herokuapp.com/predictocr',
                          json={"text_nlp[]": cropped_image.tolist(), "label": label})
        # s = address(cropped_image)




        text = s
        return text
    elif label == 'table':
        s = requests.post(url='https://ocr-tesseract-api2.herokuapp.com/predictocr',
                          json={"text_nlp[]": cropped_image.tolist(), "label": label})
        s = table(cropped_image)


        headers_table = s[0]
        values_table = s[1]
        g = []
        for i in values_table:
            f = ''.join([j for j in " ".join(i) if not j.isdigit()])
            g.append(f)
        sentence = "".join(g)
        sentence = sentence.lower()
        words = nltk.word_tokenize(sentence.lower())

        new_words = [word for word in words if word.isalnum()]
        WordSet = []
        for word in new_words:
            if word not in set(stopwords.words("english")):
                WordSet.append(word)
        final_text = " ".join(WordSet)

        category = requests.post(url='https://nlp-categorize.herokuapp.com/categorize',
                                 data={"text_nlp": str(final_text)})

        response_data = category.json()

        category_data = response_data['category']
        category_conf = response_data['category_conf']

        return jsonify({'headers':headers_table,'values':values_table})



if __name__ == '__main__':
    import argparse
    app.run(debug=False,port=5000,host='0.0.0.0')