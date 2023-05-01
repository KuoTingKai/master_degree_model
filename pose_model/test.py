import deeplabcut


if __name__ == '__main__':
    img_folder = "C:/Users/Kevin/Desktop/pose_model/img"
    config_path = "C:/Users\Kevin/Desktop/pose_model/test-aa-2021-04-28/config.yaml"


    deeplabcut.analyze_time_lapse_frames(config_path,img_folder,frametype='.png',shuffle=1,
            trainingsetindex=0,gputouse=None,save_as_csv=False)