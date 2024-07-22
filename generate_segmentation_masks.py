import argparse
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import torch

import requests
from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers import SamModel, SamProcessor

def mask_to_polygon(mask):
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon

def polygon_to_mask(polygon, image_shape):
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask

def refine_masks(masks, polygon_refinement = False):
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks

def main():
    args = argparse.ArgumentParser()
    #args.add_argument("--device", type=str, default="cuda", help="Device")
    args.add_argument("--image_dir", type=str, default="", help="Image dir")
    args.add_argument("--object_label", type=str, default="", help="Objet label for the object detection model to find")

    args = args.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load an object detection model
    processor_detr = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model_detr = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    # Load a segmentation model
    model_sam = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    processor_sam = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    # Get all the images
    images = os.listdir(args.image_dir)

    # Create a new output dir for masks
    masks_dir = args.image_dir+'_masks'
    os.mkdir(masks_dir)

    for j, image_file in enumerate(images):
        print(f'processing image {j} - {image_file}')
        # Load in the image
        image = Image.open(f'{args.image_dir}/{image_file}')
        ## Run the object detection model
        #inputs = processor_detr(images=image, return_tensors="pt")
        ##with torch.no_grad():
        #outputs = model_detr(**inputs)

        ## convert outputs (bounding boxes and class logits) to COCO API
        ## let's only keep detections with score > 0.9
        #target_sizes = torch.tensor([image.size[::-1]])
        #results = processor_detr.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        ##for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        ##    box = [round(i, 2) for i in box.tolist()]
        ##    print(
        ##            f"Detected {model_detr.config.id2label[label.item()]} with confidence "
        ##            f"{round(score.item(), 3)} at location {box}"
        ##    )

        #box_found = False
        #for i, (score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
        #    if model_detr.config.id2label[label.item()] == args.object_label:
        #        box_out = box.detach().cpu().numpy()
        #        box_out = [round(i, 2) for i in box_out.tolist()]
        #        box_found = True

        box_found = True
        if box_found:
            print(f'Found box for target label {args.object_label}')

            # Input points for segment anything to look at
            #input_points = [[[(box_out[0]+box_out[2])/2, (box_out[1]+box_out[3])/2],],]
            input_points = [[[1500,2000],],]

            # Run segmentation model
            inputs = processor_sam(image, input_points=input_points, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model_sam(**inputs)

            masks = processor_sam.image_processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
            )[0]

            masks = refine_masks(masks, True)

            mask_out = (masks[0]).astype(np.uint8)

            # Write out the mask as an image
            mask_out = Image.fromarray(mask_out)
            mask_out.save(f'{masks_dir}/mask_{image_file}')

        else:
            print(f'Box not found for target label {args.object_label} Writing empty segmentation map')

            mask_out = np.zeros(image.size).astype(np.uint8)
            mask_out = Image.fromarray(mask_out)
            mask_out.save(f'{masks_dir}/mask_{image_file}')

    print(f'Done!')

if __name__ == "__main__":
    main()
