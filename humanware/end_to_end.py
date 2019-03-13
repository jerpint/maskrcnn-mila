import sys

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision

from demo.predictor import COCODemo

sys.path.append('/home/jerpint/digit-detection/')

from trainer.utils import array_to_housenumber


class AvenueDetector(COCODemo):
    '''Class that will be used to do end to end detection'''

    def __init__(self, *args, **kwargs):
        super(AvenueDetector, self).__init__(*args, **kwargs)
        self.load_digit_detector()

    def load_digit_detector(self, model_path='/home/jerpint/digit-detection/results/SVHN_Multiloss_VGG19_trainextra_2/best_model.pth', map_location='cpu'):
        self.digit_detector = torch.load(model_path, map_location=map_location)

    CATEGORIES = ["__background", "House Number"]

    def get_img_from_bbox(self, pil_image):

        '''From a segmented image, return the only the image of the bbox'''

        # convert to BGR format
        image = np.array(pil_image)[:, :, [2, 1, 0]]
        predictions = self.compute_prediction(image)
        best_predictions = self.select_top_predictions(predictions)
        bbox = best_predictions.bbox[0].cpu().numpy()
        bbox_round = np.round(bbox).astype('int')
        img_from_bbox = image[bbox_round[1]:bbox_round[3], bbox_round[0]:bbox_round[2], :]

        # convert to RGB format
        img_from_bbox = np.array(img_from_bbox)[:, :, [0, 1, 2]]
        image_bbox_pil = Image.fromarray(img_from_bbox.astype('uint8'), 'RGB')

        return image_bbox_pil, predictions

    def extract_house_number(self, image_bbox_pil):
        '''
        Get the house number using the digit detector trained on SVHN.

        '''

        img_tensor = self.digit_detection_preprocess(image_bbox_pil)
        outputs = self.digit_detector.forward(img_tensor)
        batch_preds = []

        for index in range(len(outputs)):
            pred = outputs[index]
            _, predicted = torch.max(pred.data, 1)
            batch_preds.append(predicted)

        batch_preds = torch.stack(batch_preds)  # Combine all results to one tensor
        batch_preds = batch_preds.transpose(1, 0)  # Get same shape as target
        batch_preds = batch_preds.cpu().numpy().astype('int') # Convert to array
        house_number = array_to_housenumber(batch_preds)
        return house_number

    def digit_detection_preprocess(self, pil_image):
        '''
        Perform preprocessing of the image to be used by the digit
        detector.

        Arguments:
            pil_image : PIL Image object
                image object of the contents of the bounding box
        '''

        resize = torchvision.transforms.Resize((54, 54), interpolation=2)
        resized_img = np.asarray(resize(pil_image))

        assert resized_img.shape == (54, 54, 3)

        # mean and std per channel when trained on SVHN
        images_mean = (109.7994, 110.00522, 114.33739)
        images_std = (12.675092, 12.741672, 11.369844)

        img_normalized = (resized_img - images_mean) / images_std

        # swap color axis
        # numpy image: H x W x C
        # torch image: C X H X W
        img_normalized = img_normalized.transpose((2, 0, 1))
        img_tensor = torch.from_numpy(img_normalized).float()
        img_tensor = img_tensor[None, :, :, :]

        return img_tensor

    def overlay_class_names(self, image, predictions):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        box_pil_img, predictions = self.get_img_from_bbox(image)
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        labels = [self.CATEGORIES[i] for i in labels]
        boxes = predictions.bbox
        house_number = self.extract_house_number(box_pil_img)

        template = "{}: {:d} , bbox conf: {:.2f}"
        for box, score, label, hn in zip(boxes, scores, labels, house_number):
            x, y = box[:2]
            s = template.format(label, hn, score)
            cv2.putText(
                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
            )

        return image, predictions, house_number
