from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

model = YOLO("creative_fox.pt")
img_path = "val/e163da3cac840bda529512.jpg"

results = model.predict(img_path)

img = Image.open(img_path)
plt.imshow(img)

for result in results:
    
    boxes = result.boxes.xyxy 
    
    for box in boxes:
        x1, y1, x2, y2 = box.tolist()
        
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                          edgecolor='r', facecolor='none', linewidth=2))

plt.show()
