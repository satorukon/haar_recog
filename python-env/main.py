from __future__ import division

import cv2
import time


# 評価器を読み込み
# https://github.com/opencv/opencv/tree/master/data/haarcascades
cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye_tree_eyeglasses.xml')



def mosaic(src, ratio=0.1):
    small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

def mosaic_area(src, x, y, width, height, ratio=0.1):
    dst = src.copy()
    dst[y:y + height, x:x + width] = mosaic(dst[y:y + height, x:x + width], ratio)
    return dst



#カメラインスタンス作成
#cap = cv2.VideoCapture(0)
URL = "rtsp://rtsp:nutanix@192.168.199.196:8554/ipcam_h264.sdp"
#URL = "rtsp://rtsp:nutanix@192.168.200.103:8554/ipcam_h264.sdp"
cap = cv2.VideoCapture(URL)

assert cap.isOpened(), 'Cannot capture source'

frames = 0
start = time.time()   
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            #resize_frame = cv2.resize(frame, (int(frame.shape[1] * 0.8), int(frame.shape[0] * 0.8)))
            resize_frame = frame
            #cv2.imshow("frame", resize_frame)

            #検出用にGray化
            gray = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2GRAY)
            #cv2.imshow("gray", gray)


            # 顔検出
            #物体認識（顔認識）の実行
            #image – CV_8U 型の行列．ここに格納されている画像中から物体が検出されます
            #objects – 矩形を要素とするベクトル．それぞれの矩形は，検出した物体を含みます
            #scaleFactor – 各画像スケールにおける縮小量を表します
            #minNeighbors – 物体候補となる矩形は，最低でもこの数だけの近傍矩形を含む必要があります
            #flags – このパラメータは，新しいカスケードでは利用されません．古いカスケードに対しては，cvHaarDetectObjects 関数の場合と同じ意味を持ちます
            #minSize – 物体が取り得る最小サイズ．これよりも小さい物体は無視されます
            facerect = cascade.detectMultiScale(
                gray,
                scaleFactor=1.11,
                minNeighbors=3,
                minSize=(100, 100)
            )
   
            if len(facerect) != 0:
                for x, y, w, h in facerect:
                    cv2.rectangle(resize_frame, (x,y), (x+w, y+h), (255,0,0), 5)
                    # 顔の部分(この顔の部分に対して目の検出をかける)
                    face_gray = gray[y: y + h, x: x + w]
                    # 顔の部分から目の検出
                    eyes = eye_cascade.detectMultiScale(
                        face_gray,
                        scaleFactor=1.11, # ここの値はPCのスペックに依存するので適宜修正してください
                        minNeighbors=3,
                        minSize=(15, 15)
                    )

                    if len(eyes) == 0:
                        # 目が閉じられたとみなす
                        
                        cv2.putText(
                            resize_frame,
                            'close eyes',
                            (x, y), # 位置を少し調整
                            cv2.FONT_HERSHEY_PLAIN,
                            2,
                            (0, 255,0),
                            2,
                            cv2.LINE_AA
                        )
                        
                    else:
                        for (ex, ey, ew, eh) in eyes:
                            # 目の部分にモザイク処理
                            ex = int((x + ex) - ew / 2)
                            ey = int(y + ey)
                            ew = int(ew * 2.5)
                            resize_frame = mosaic_area(resize_frame,ex,ey,ew,eh)
                            cv2.rectangle(resize_frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

 







            # 何か処理（ここでは文字列「hogehoge」を表示する）
            edframe = resize_frame
            cv2.putText(edframe, 'hogehoge', (0,50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255,0), 3, cv2.LINE_AA)
            # 加工済の画像を表示する
            cv2.imshow('Edited Frame', edframe)


            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))

        else:
            break


except KeyboardInterrupt: # except the program gets interrupted by Ctrl+C on the keyboard.
    print("\nCamera Interrupt")

finally:
    cap.release()
    cv2.destroyAllWindows()
