# Color Equalization with Gamma Correction, Segmentation Based on Color and Localization of Patch in a Image

## Controls

After changing any value in the trackbar, any key must be pressed.

"Esc" to exit the program.

## Global Trackbar:

task(0-2) - change task

# Color Equalization and Gamma Correction:

#### Trackbars

image(0-1) - change image

gamma(1-100) - change gamma value

#### Procedure for each color model:

1. Change the image from BGR to the specified color model.

2. Split the image into channels.

3. Display the histogram of the distribution of individual channels.

4. Equalize the individual channels.

5. Display the histogram of the distribution and individual channels after equalization.

6. Apply gamma correction according to the selected value on the trackbar and adjust the displayed channels.

## Implementation of Gamma Correction

The trackbar values are divided by 25 (to obtain values less than 1).

A calculated LUT (Look Up Table), which contains the adjusted value for each possible pixel value.

Adjustment of the image based on the LUT


### Grayscale

![image](https://user-images.githubusercontent.com/55833503/227801828-9e729197-ccb7-47c0-8cf6-649ef35c6864.png)

#### Before equalization

![image](https://user-images.githubusercontent.com/55833503/227802070-e9ae3759-bab5-4f2d-9dd4-7f8086a02e06.png)

#### After equalization

![image](https://user-images.githubusercontent.com/55833503/227802077-1a9a755f-14f5-41a4-be08-c47041a71def.png)

### RGB (gamma = 0.8)

![image](https://user-images.githubusercontent.com/55833503/227801844-17ae8006-6d24-417d-a5a7-18ba0605ee3f.png)

#### Before equalization

![image](https://user-images.githubusercontent.com/55833503/227802262-e2f27cac-af0a-4ea2-bdc3-4ef509b2e24b.png)

#### After equalization

![image](https://user-images.githubusercontent.com/55833503/227802272-f61b5dab-ee56-46d8-9bbf-47ef616dd5be.png)

### YCbCr  (gamma = 0.8)

![image](https://user-images.githubusercontent.com/55833503/227801916-b90490c6-bcb6-4d3e-a84c-2fbf1c49c4b8.png)

#### Before equalization

![image](https://user-images.githubusercontent.com/55833503/227802282-5371c655-7bb6-427e-ad73-9a2eec8badc2.png)

#### After equalization

![image](https://user-images.githubusercontent.com/55833503/227802320-0b5468e2-a484-4ca3-a907-b519b844c08c.png)

### HSV (gamma = 0.8)

![image](https://user-images.githubusercontent.com/55833503/227801935-043255f1-8e8f-4ae9-b0ad-2fdef8cdea2d.png)

#### Before equalization

![image](https://user-images.githubusercontent.com/55833503/227802393-ca291857-b6d3-48ae-8d3e-40640de8642e.png)

#### After equalization

![image](https://user-images.githubusercontent.com/55833503/227802403-01a50180-3d81-45c6-9347-acf8c6d6d557.png)

### XYZ (gamma = 0.8)

![image](https://user-images.githubusercontent.com/55833503/227801975-dbae7073-839f-4d51-8994-a275a120429e.png)

#### Before equalization

![image](https://user-images.githubusercontent.com/55833503/227802407-3d8fb356-652d-47fd-8f2d-3c12a283f80a.png)

#### After equalization

![image](https://user-images.githubusercontent.com/55833503/227802422-3a7a571e-b322-4fd2-a409-f0520290c7d1.png)

### Lab (gamma = 0.8)

![image](https://user-images.githubusercontent.com/55833503/227802000-3c106c15-5ee8-43c8-862e-6a42882e7e31.png)

#### Before equalization

![image](https://user-images.githubusercontent.com/55833503/227802434-ddf815df-86d0-4be2-a139-b1b5dfde4291.png)

#### After equalization

![image](https://user-images.githubusercontent.com/55833503/227802485-262a01d8-3a97-41eb-a70f-342546ac364d.png)

### Image 2 - option to change the image in the "trackbars" window.

# Segmentation Based on Color

#### Trackbars

Image (0-1) - change image

thresh_min (0-255) - adjust minimum threshold value

Procedure:

1. Correctly convert the image to the Lab model (to float and then to Lab).

2. Cut out a smaller section of the image (8x8) that contains the segmented color.

3. Calculate the average color from the smaller section (average of individual channels).

4. Calculate the Euclidean distance from the average color for each pixel in the image.

5. Based on the distance, create a grayscale image (each pixel contains the distance from the average color).

6. Invert the pixel values for easier application of the mask (255 being closest, 0 being farthest).

7. Apply binary thresholding to filter only colors close to the average color (create a mask).

8. Apply the mask to the original image.

## Image 1

### Cropped section displayed in RGB (Lab used for calculation)

![image](https://user-images.githubusercontent.com/55833503/227802814-63042016-bff5-4c08-8c1b-c144114c89ef.png)

### Image with distances

![image](https://user-images.githubusercontent.com/55833503/227802825-31f64524-242c-4f77-ab6e-790f3f9741dc.png)

### Mask

![image](https://user-images.githubusercontent.com/55833503/227802837-4f2fe3ff-5e8c-4542-82f2-769dcf5ea515.png)

### Image after applying the mask

![image](https://user-images.githubusercontent.com/55833503/227802853-3ecec79f-dfb4-4865-9b97-13abdf01ad8c.png)

## Image 2

### Cropped section displayed in RGB (Lab used for calculation)

![image](https://user-images.githubusercontent.com/55833503/227802903-9d51709b-0cd5-4af7-a9cc-9254fc0280d5.png)

### Image with distances

![image](https://user-images.githubusercontent.com/55833503/227802916-7e41384a-4f1d-48b6-8659-89d71bd37c09.png)

### Mask

![image](https://user-images.githubusercontent.com/55833503/227802926-023232d9-be9a-4bc9-82c2-9533d6948ced.png)

### Image after applying the mask

![image](https://user-images.githubusercontent.com/55833503/227802939-f0d4587a-b752-46bd-ad3f-67ec5ae6ffde.png)

# Localization of Patch in the Image

#### Trackbars

detector(0-3) - change detector and descriptor:

0 - SIFT + SIFT

1 - FAST + SIFT

2 - ORB + ORB

3 - Harris + SIFT

Procedure:
1. Apply the detector to all images (main + patches).

2. Apply the descriptor to all images.

3. Find matches between different combinations of patches and the main image.

4. Display the 50 best matches for each patch (for clarity of images).

In the images, we can see that the most successful combination is SIFT and SIFT. Other combinations very inaccurately localized the position of patches in the main image.

## SIFT detector + SIFT descriptor

### Patch 1

![image](https://user-images.githubusercontent.com/55833503/227970767-5a3cab68-d952-4fa5-9f6c-f7236ed8a33e.png)

### Patch 2

![image](https://user-images.githubusercontent.com/55833503/227971407-33ba5897-4d00-438d-b65d-49d9147e8977.png)

### Patch 3

![image](https://user-images.githubusercontent.com/55833503/227971515-137ee3de-a34c-4eb9-9729-d8b7c6dfcfcb.png)

## FAST detector + SIFT descriptor

### Patch 1

![image](https://user-images.githubusercontent.com/55833503/227971681-9f659059-e425-4b4f-9f55-401a0fac31c6.png)

### Patch 2

![image](https://user-images.githubusercontent.com/55833503/227971756-d608f5c9-64dd-4e88-b10c-eaffe7af59c9.png)

### Patch 3

![image](https://user-images.githubusercontent.com/55833503/227971870-57f03720-928c-4f2a-bd05-1c9040b2bdad.png)

## ORB detector + ORB descriptor

### Patch 1

![image](https://user-images.githubusercontent.com/55833503/227972038-379b5ec8-ca2c-4f5f-8a19-70c6f56f807c.png)

### Patch 2

![image](https://user-images.githubusercontent.com/55833503/227972142-16dc4cde-bcd4-4af4-aed5-fc802897183e.png)

### Patch 3

![image](https://user-images.githubusercontent.com/55833503/227972250-bcfa0be8-d9be-457c-9875-7ff0ac12d96b.png)

## Harris detector + SIFT descriptor

### Patch 1

![image](https://user-images.githubusercontent.com/55833503/227973586-e5bc7c12-3f33-4951-9d2b-a489661a09d1.png)

### Patch 2

![image](https://user-images.githubusercontent.com/55833503/227973659-fe4f3c43-911a-4347-8903-7816864a317d.png)

### Patch 3

![image](https://user-images.githubusercontent.com/55833503/227973724-04ce58c4-3f62-4c51-89e9-1865cb5e64f6.png)

