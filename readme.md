/process_data.py : holistic_json data를 전처리하는 함수. json_Files를 불러와서, 라벨과 함께 로드 후 자동으로 feature extraction(피쳐추출)하고 /dataset에 X_total,Y_total 저장


/train_mlp_classifier : 주어진 데이터와 모델을가지고 학습을 하는 파일. 
      18 line : /classifier 안에있는 classifier을 로드 (classifier 바꿔서 로드하려면 이름 바꿔서 실행하면 됨 {name}_classifier.py)
      학습 후 자동으로 /model 에 weight파일, v1.pt가 추가됨. /report에도 추가됨
      또한 /realtime_pose_predictor/classifier에도 classifer가 자동으로 추가되고 /realtime_pose_predictor/model에도 v1.pt(weight)이 자동으로 들어가므로
      학습된 데이터를 어플리케이션으로 테스트하고싶으면 학습후 바로 /realtime_pose_predictor/app.py 실행하면 테스트 가능

      
/classifier
      딥러닝 모델 : 추가해가면서 테스트하면 됨. 파일 이름은 {name}_classifier.py 로 고정(자동으로 매칭하기위해서 이름을 고정시켜야함)

      
/model
      학습된 .pt파일이 저장(weight)


/tsne.py : 현재 train용 데이터셋을 2d 시각화


/train_mlp_classifier/tsne.py :
          /train_mlp_classifier/app.py로 테스트 종료후, 데이터가 자동으로 저장되는데 이때 수집한 데이터들을 2d로 시각화
