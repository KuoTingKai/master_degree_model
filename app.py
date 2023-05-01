from flask import Flask, render_template, request, redirect, url_for,jsonify
import pathlib
import os
from werkzeug.utils import secure_filename
import deeplabcut
import shutil
from pose_model import depth_comparison_feature_single_model as dp_model
import os

app_folder = pathlib.Path(__file__).parent.absolute()
upload_folder = os.path.join(app_folder,  'static', 'uploads')


app = Flask(__name__)
app.config["ENV"] = "DEEPLABCUT"
app.config['JSON_AS_ASCII'] = False




@app.route("/", methods=['GET'])
def index():
    return render_template('index.html')

@app.route("/", methods=['POST'])
def upload_file():

    if request.form.get('submit_name') == 'image_upload':
        files = request.files.getlist("img_name")
        # print(files)
        if files.count:
            for file in files:
                file.save(os.path.join(upload_folder, secure_filename(file.filename)))
        run_deeplabcut(request)
        result = dp_model.analize_image(app_folder)
        result = transfer_result_str(result)
        remove(upload_folder)
        return jsonify(result=list(result))
    else:
        files = request.files.getlist("viedo_name")
        # print(files)
        if files.count:
            for file in files:
                name = file.filename
                file.save(os.path.join(upload_folder, secure_filename(file.filename)))
        run_deeplabcut(request,name)
        

    return redirect(url_for('index'))



def run_deeplabcut(request,name=None):

    config = 'pose_model/0611_depth_2021_06_12/config.yaml'
    config = os.path.join(app_folder, config)

    if request.form.get('submit_name') == 'image_upload':
        deeplabcut.analyze_time_lapse_frames(config,upload_folder,frametype='.png',shuffle=4,
                trainingsetindex=0,gputouse=None,save_as_csv=True)
    else:
        # files = request.files.getlist("viedo_name")
        deeplabcut.analyze_videos(config, os.path.join(upload_folder,name), videotype='avi', shuffle=4,
                                   trainingsetindex=0, gputouse=0, save_as_csv=True)


def remove(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def transfer_result_str(result):
    str = []
    for i in result:
        if i == '1':
            str.append('健康')
        else:
            str.append('疼痛')
    return str



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug = True, port=5000)