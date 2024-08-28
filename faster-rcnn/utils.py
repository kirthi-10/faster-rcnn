import torch
from PIL import Image
from torchvision import transforms
import os

# def get_prediction(img_path, model, threshold=0.75):
#     img = Image.open(img_path)
#     transform = transforms.Compose([transforms.ToTensor()])
#     img = transform(img).unsqueeze(0)  # Add batch dimension
#     with torch.no_grad():  # Disable gradient calculation for inference
#         pred = model(img)
#     pred_class = [model.idx_to_class[i] for i in list(pred[0]['labels'].numpy())]
#     pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
#     pred_score = list(pred[0]['scores'].detach().numpy())
    
#     # Filter results based on threshold
#     pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
#     pred_boxes = pred_boxes[:pred_t+1]
#     pred_class = pred_class[:pred_t+1]
#     return pred_boxes, pred_class


from PIL import ImageDraw, Image, ImageFont

def draw_bounding_boxes(img_path, prediction):
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    boxes = prediction[0]['boxes'].cpu().numpy()  # Bounding boxes
    labels = prediction[0]['labels'].cpu().numpy()  # Class labels
    scores = prediction[0]['scores'].cpu().numpy()  # Confidence scores
    idx_to_classes = {0: 'person', 1: 'chair', 2: 'car', 3: 'dog', 4: 'bottle', 5: 'tvmonitor', 6: 'aeroplane', 
                      7: 'cat', 8: 'sheep', 9: 'motorbike', 10: 'sofa', 11: 'pottedplant', 12: 'horse', 
                      13: 'car', 14: 'bicycle', 15: 'cow', 16: 'train', 17: 'bus', 18: 'boat', 19: 'diningtable', 
                      20: 'bird'}


    for box, label, score in zip(boxes, labels, scores):
        if score < 0.79 :
            continue
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
        text = f"{idx_to_classes[label]} {score:.2f}"  # Class name and confidence score
        print(text)
        font = ImageFont.load_default()  
        draw.text((box[0] + 2, box[1] + 2), text, font=font, fill="blue")
    


    print(img_path)
    # Create the result directory if it doesn't exist
    result_dir = "static/uploads/results/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Create the result path
    result_img_path = os.path.join(result_dir, os.path.basename(img_path))
    
    print(f'Saving image to: {result_img_path}')
    
    try:
        image.save(result_img_path)
        print(f'Image saved successfully to: {result_img_path}')
    except Exception as e:
        print(f'Error saving image: {e}')
    
    print('path:')
    print(result_img_path)

    return result_img_path[7:]