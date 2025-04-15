#pip install ultralytics
#pip install --upgrade ultralytics

from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('')

assert cap.isOpened(),'Erro ao processar vídeo'
w,h,fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

classes_to_count = [2]

region_points = [(20,400), (1080,404), (1080, 360), (20, 360)]

video_writer = cv2.VideoWriter('Object_couting_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps,(w,h))

counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
                 reg_pts=region_points,
                 classes_names=model.names,
                 draw_tracks=True)

while cap.isOpened():
    sucess, im0 = cap.read()
    if not sucess:
        print('O frame do vídeo está vazio ou processado com sucesso')
        break
    tracks = model.track(im0, persist=True, show=False, classes=classes_to_count)

    im0 = counter.start_counting(im0, tracks)
    video_writer.write(im0)

    if cv2.waitKey(5)&0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()