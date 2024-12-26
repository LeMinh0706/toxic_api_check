from ultralytics import YOLOv10


def main():
    
    model = YOLOv10('best.pt')
    source_path = 'uploads/images.jpg'
    results = model(source = source_path, conf = 0.25, save = False)
    class_id=6
    class_label="None"
    
    for box in results[0].boxes:
            class_id = int(box.cls)  
            class_label = results[0].names[class_id] 

    

    print(f'\nDetected class: {class_label}')
    print(f'\nDetected classid: {class_id}')
if __name__ == "__main__":
    main()


