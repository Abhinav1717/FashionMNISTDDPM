import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as T
import numpy as np 
def plot_comparison(predicted_annotations,image,gt_annotations,log_writer,iteration_no):
    
    transform = T.ToPILImage()
    image = transform(image)
    image = np.asarray(image)
    predicted_annotations = predicted_annotations.tolist()
    gt_annotations = [int(x) for x in gt_annotations]
    predicted_annotations = [int(x) for x in predicted_annotations]
    
    image = cv2.rectangle(image,(predicted_annotations[0],predicted_annotations[1]),(predicted_annotations[2],predicted_annotations[3]),(255,0,0),1)        #Draws rectangle using the top left and bottom right corner points specified
    image = cv2.rectangle(image,(gt_annotations[0],gt_annotations[1]),(gt_annotations[2],gt_annotations[3]),(0,0,255),1)        #Draws rectangle using the top left and bottom right corner points specified

    log_writer.add_image("Training",image,iteration_no,dataformats='HWC')
    plt.close()