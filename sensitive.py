def sensitive(yolo, location):

    results = yolo(source = location, conf = 0.25, save = False)
    class_id=6
    class_label=""

    for box in results[0].boxes:
            class_id = int(box.cls)  
            class_label = results[0].names[class_id] 
    
    return class_id, class_label