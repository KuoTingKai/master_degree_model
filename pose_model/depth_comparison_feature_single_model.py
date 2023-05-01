import os 
import csv
from sklearn.svm import SVC
import cv2
from .rotate import head_rotate as head_R
from .rotate import mid_body_rotate as mid_body_R
from .rotate import body_tail_rotate as body_tail_R
import deeplabcut
from sklearn import preprocessing
import joblib

def depth_value(image):
    R = image[:,:,2]
    G = image[:,:,1]
    B = image[:,:,0]

    depth = B * 256 + R
    depth = depth.astype('int16')
    return depth

def csv_read(filename):
    with open(filename, newline='') as csvfile:
  # 以冒號分隔欄位，讀取檔案內容
        rows = csv.reader(csvfile)
        data = []
        count = 0
        for row in rows:
            if count < 3:
                count = count + 1
                continue
            if (float(row[1]) or float(row[4]) or float(row[7]) or float(row[10]) or float(row[13]) or float(row[16])) > 1100:
                continue
            elif (float(row[1]) or float(row[4]) or float(row[7]) or float(row[10]) or float(row[13]) or float(row[16])) < 600:
                continue
            elif (float(row[2]) or float(row[5]) or float(row[8]) or float(row[11]) or float(row[14]) or float(row[17])) > 800:
                continue
            elif (float(row[2]) or float(row[5]) or float(row[8]) or float(row[11]) or float(row[14]) or float(row[17])) < 400:
                continue

            a = [float(row[1]),float(row[2])]
            b = [float(row[4]),float(row[5])]
            c = [float(row[7]),float(row[8])]
            d = [float(row[10]),float(row[11])]
            e = [float(row[13]),float(row[14])]
            f = [float(row[16]),float(row[17])]

            data.append([a,b,c,d,e,f])

    return data



def analysis_video(config_path,img_folder):
    # deeplabcut.analyze_videos(config_path, ['fullpath/analysis/project/videos/reachingvideo1.avi'], save_as_csv=True)
    deeplabcut.analyze_time_lapse_frames(config_path,img_folder,frametype='.png',shuffle=1,
                trainingsetindex=0,gputouse=None,save_as_csv=False,rgb=True)


def analize_image(app_folder):
    box = 10
    test_csv = os.path.join(app_folder,'static/uploads/uploadsDLC_resnet50_0611June12shuffle4_200000.csv')
    img_folder = os.path.join(app_folder,'static/uploads')
    test_GT = []
    img_list = [_ for _ in os.listdir(img_folder) if _.endswith('.png')]
    img_list.sort(key=lambda x:int(x[5:-4]))

    test_feature = []

    with open(test_csv, newline='') as csvfile:
        count = 0
        rows = csv.reader(csvfile)
        for row in rows:
            count = count + 1
            if count > 3:
                test_GT.append(row)
            else:
                continue
    
    lst = [0] * 60
    for i in range(len(img_list)):
        feature = []
        img_name = os.path.join(img_folder,img_list[i])
        img = cv2.imread(img_name)
        depth_img = depth_value(img)
        abcd,abcd_angle = head_R(depth_img,test_GT[i],box)
        e,e_angle = mid_body_R(depth_img,test_GT[i],box)
        f,f_angle = body_tail_R(depth_img,test_GT[i],box)
        feature.extend(abcd)
        feature.extend(e)
        feature.extend(f)
        feature.extend(lst)
        test_feature.append(feature)

    min_max_scale = preprocessing.MinMaxScaler()
    X = min_max_scale.fit_transform(test_feature)

    loaded_model = joblib.load(os.path.join(app_folder, 'pose_model/SVM_model/depth/MLW.train'))
    result = loaded_model.predict(X)
    print(result)
    return result

# if __name__ == '__main__':
#     analize_image()
    