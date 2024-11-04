import cv2
from matplotlib import pyplot as plt
import numpy as np

def nothing(x):
    pass


def resize_image_ratio(img, ratio_percent):
    width = int(img.shape[1] * ratio_percent / 100)
    height = int(img.shape[0] * ratio_percent / 100)
    dim = (width, height)

    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return img

def create_image_histogram_3_channel(img, name):
    plt.figure()
    plt.title(name)
    colors = ('b', 'g', 'r')
    for i in range(0, 3):
        histogram = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histogram, color=colors[i])
    plt.show()


def create_image_histogram_1_channel(img, name):
    plt.figure()
    plt.title(name)
    histogram = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(histogram, color='r')
    plt.show()


def combine_pictures(images, axis):
    final_image = resize_image_ratio(images[0], 10)
    for i in range(1, len(images)):
        final_image = np.concatenate((final_image, resize_image_ratio(images[i], 10)), axis=axis)

    return final_image


def equalize_rgb(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    create_image_histogram_3_channel(img_rgb, "before equalization RGB")

    img_rgb[:, :, 0] = cv2.equalizeHist(img_rgb[:, :, 0])
    img_rgb[:, :, 1] = cv2.equalizeHist(img_rgb[:, :, 1])
    img_rgb[:, :, 2] = cv2.equalizeHist(img_rgb[:, :, 2])

    channel_R = img_rgb[:, :, 0]
    channel_G = img_rgb[:, :, 1]
    channel_B = img_rgb[:, :, 2]

    create_image_histogram_3_channel(img_rgb, "after equalization RGB")

    final_image = combine_pictures([channel_R, channel_G, channel_B], 1)

    return final_image


def equalize_xyz(img):
    img_xyz = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)

    create_image_histogram_3_channel(img_xyz, "before equalization XYZ")

    img_xyz[:, :, 0] = cv2.equalizeHist(img_xyz[:, :, 0])
    img_xyz[:, :, 1] = cv2.equalizeHist(img_xyz[:, :, 1])
    img_xyz[:, :, 2] = cv2.equalizeHist(img_xyz[:, :, 2])

    channel_X = img_xyz[:, :, 0]
    channel_Y = img_xyz[:, :, 1]
    channel_Z = img_xyz[:, :, 2]

    create_image_histogram_3_channel(img_xyz, "after equalization XYZ")

    final_image = combine_pictures([channel_X, channel_Y, channel_Z], 1)

    return final_image



def equalize_ycrcb(img):
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    create_image_histogram_3_channel(img_ycrcb, "before equalization YCrCb")

    img_ycrcb[:, :, 0] = cv2.equalizeHist(img_ycrcb[:, :, 0])
    img_ycrcb[:, :, 1] = cv2.equalizeHist(img_ycrcb[:, :, 1])
    img_ycrcb[:, :, 2] = cv2.equalizeHist(img_ycrcb[:, :, 2])

    channel_Y = img_ycrcb[:, :, 0]
    channel_Cr = img_ycrcb[:, :, 1]
    channel_Cb = img_ycrcb[:, :, 2]

    create_image_histogram_3_channel(img_ycrcb, "after equalization YCrCb")

    final_image = combine_pictures([channel_Y, channel_Cr, channel_Cb], 1)

    return final_image


def equalize_hsv(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    create_image_histogram_3_channel(img_hsv, "before equalization HSV")

    img_hsv[:, :, 0] = cv2.equalizeHist(img_hsv[:, :, 0])
    img_hsv[:, :, 1] = cv2.equalizeHist(img_hsv[:, :, 1])
    img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])

    create_image_histogram_3_channel(img_hsv, "after equalization HSV")

    channel_h = img_hsv[:, :, 0]
    channel_s = img_hsv[:, :, 1]
    channel_v = img_hsv[:, :, 2]

    final_image = combine_pictures([channel_h, channel_s, channel_v], 1)

    return final_image


def equalize_gray(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    create_image_histogram_1_channel(img_gray, "before equalization GRAY")

    img_gray = cv2.equalizeHist(img_gray)

    create_image_histogram_1_channel(img_gray, "after equalization GRAY")

    cv2.imshow('after equalization GRAY', resize_image_ratio(img_gray, 10))

    return img_gray


def equalize_lab(img):
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    create_image_histogram_3_channel(img_lab, "before equalization Lab")

    img_lab[:, :, 0] = cv2.equalizeHist(img_lab[:, :, 0])
    img_lab[:, :, 1] = cv2.equalizeHist(img_lab[:, :, 1])
    img_lab[:, :, 2] = cv2.equalizeHist(img_lab[:, :, 2])

    create_image_histogram_3_channel(img_lab, "after equalization Lab")

    channel_L = img_lab[:, :, 0]
    channel_a = img_lab[:, :, 1]
    channel_b = img_lab[:, :, 2]

    final_image = combine_pictures([channel_L, channel_a, channel_b], 1)

    return final_image


def gamma_correction(img, gamma):
    gamma = 1 / gamma

    table = [((i / 255) ** gamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(img, table)


def equalization_create_windows():
    cv2.namedWindow('trackbars', cv2.WINDOW_NORMAL)

    cv2.resizeWindow("trackbars", 700, 300)

    cv2.createTrackbar('TASK 1-3', 'trackbars', 0, 2, nothing)

    cv2.createTrackbar('IMG', 'trackbars', 0, 1, nothing)

    cv2.createTrackbar('gamma_correction', 'trackbars', 20, 100, nothing)

    cv2.setTrackbarMin('gamma_correction', 'trackbars', 1)

def equalization_task():

    if cv2.getTrackbarPos('IMG', 'trackbars') == 0:
        img_path = "image_1.png"
    else:
        img_path = "image_2.png"

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    gamma = cv2.getTrackbarPos('gamma_correction', 'trackbars')
    gamma = gamma / 25

    equalize_gray(img)

    cv2.imshow('after equalization RGB', gamma_correction(equalize_rgb(img), gamma))
    cv2.imshow('after equalization YCrCb', gamma_correction(equalize_ycrcb(img), gamma))
    cv2.imshow('after equalization HSV', gamma_correction(equalize_hsv(img), gamma))
    cv2.imshow('after equalization XYZ', gamma_correction(equalize_xyz(img), gamma))
    cv2.imshow('after equalization Lab', gamma_correction(equalize_lab(img), gamma))



def lab_transfer(img):
    float_img = img.astype(np.float32)

    float_img /= 255

    float_img_lab = cv2.cvtColor(float_img, cv2.COLOR_BGR2LAB)

    return float_img_lab


def average_color_crop_image(img, x, y, mat_size):
    crop_img = img[y:y + mat_size, x: x + mat_size]
    color = [0, 0, 0]
    color[0] = np.mean(crop_img[:, :, 0])
    color[1] = np.mean(crop_img[:, :, 1])
    color[2] = np.mean(crop_img[:, :, 2])

    return crop_img, color


def compute_distance(img1, color):

    img2 = np.zeros((img1.shape[0], img1.shape[1], 3), np.float32)

    img2[:] = color
    inverse = 255

    diff_img = cv2.add(img1, -img2)

    diff_L = diff_img[:, :, 0]
    diff_A = diff_img[:, :, 1]
    diff_B = diff_img[:, :, 2]

    delta_e_img = np.sqrt(diff_L * diff_L + diff_A * diff_A + diff_B * diff_B)

    return inverse - delta_e_img.astype(np.uint8)


def color_segmentation_create_windows():
    cv2.namedWindow('trackbars', cv2.WINDOW_NORMAL)

    cv2.resizeWindow("trackbars", 700, 300)

    cv2.createTrackbar('TASK 1-3', 'trackbars', 1, 2, nothing)

    cv2.createTrackbar('IMG', 'trackbars', 0, 1, nothing)

    cv2.createTrackbar('thresh_min', 'trackbars', 240, 255, nothing)


def color_segmentation():

    if cv2.getTrackbarPos('IMG', 'trackbars') == 0:
        img_path = "image_1.png"
        patch_postion_x, patch_postion_y = 1793, 2775
    else:
        img_path = "image_2.png"
        patch_postion_x, patch_postion_y = 2030, 3269

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    img_lab = lab_transfer(img)

    color = average_color_crop_image(img_lab, patch_postion_x, patch_postion_y, 8)[1]

    cv2.imshow('Cropped image', resize_image_ratio(average_color_crop_image(img, patch_postion_x, patch_postion_y, 8)[0], 10000))

    delta_E_img = compute_distance(img_lab, color)
    cv2.imshow('Distance', resize_image_ratio(delta_E_img, 10))

    thresh_min = cv2.getTrackbarPos('thresh_min', 'trackbars')

    ret, threshold = cv2.threshold(delta_E_img, thresh_min, 255, cv2.THRESH_BINARY)

    cv2.imshow('Tresholded mask', resize_image_ratio(threshold, 10))

    masked_img = cv2.bitwise_and(img, img, mask=threshold)

    cv2.imshow('Masked img', resize_image_ratio(masked_img, 10))


def detect_descript_create_windows():
    cv2.namedWindow('trackbars', cv2.WINDOW_NORMAL)

    cv2.resizeWindow("trackbars", 700, 300)

    cv2.createTrackbar('TASK 1-3', 'trackbars', 2, 2, nothing)

    cv2.createTrackbar('DETECTOR', 'trackbars', 0, 3, nothing)


def harris(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    img_corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    result_img = img.copy()
    result_img[img_corners > 0.05 * img_corners.max()] = [0, 0, 255]
    keypoints = np.argwhere(img_corners > 0.05 * img_corners.max())
    keypoints = [cv2.KeyPoint(float(x[1]), float(x[0]), 13) for x in keypoints]
    return (keypoints, result_img)


def use_detectors_descriptors():

    img = cv2.imread("local_descriptors_task/lookup.tif", cv2.IMREAD_COLOR)

    patches = []

    patches.append(cv2.imread("local_descriptors_task/patch1.tif", cv2.IMREAD_COLOR))
    patches.append(cv2.imread("local_descriptors_task/patch2.tif", cv2.IMREAD_COLOR))
    patches.append(cv2.imread("local_descriptors_task/patch3.tif", cv2.IMREAD_COLOR))

    detector = cv2.getTrackbarPos('DETECTOR', 'trackbars')

    if detector == 0:
        sift = cv2.SIFT_create()
        kp = sift.detect(img, None)

        kp1 = sift.detect(patches[0], None)
        kp2 = sift.detect(patches[1], None)
        kp3 = sift.detect(patches[2], None)

    elif detector == 1:
        fast = cv2.FastFeatureDetector_create()
        kp = fast.detect(img, None)

        kp1 = fast.detect(patches[0], None)
        kp2 = fast.detect(patches[1], None)
        kp3 = fast.detect(patches[2], None)

    elif detector == 2:
        orb = cv2.ORB_create()
        kp = orb.detect(img, None)

        kp1 = orb.detect(patches[0], None)
        kp2 = orb.detect(patches[1], None)
        kp3 = orb.detect(patches[2], None)

    else:
        kp, img = harris(img)

        kp1, patches[0] = harris(patches[0])
        kp2, patches[1] = harris(patches[1])
        kp3, patches[2] = harris(patches[2])

    if detector == 2:
        orb = cv2.ORB_create()
        kp, des = orb.compute(img, kp)

        kp1, des1 = orb.compute(patches[0], kp1)
        kp2, des2 = orb.compute(patches[1], kp2)
        kp3, des3 = orb.compute(patches[2], kp3)

    else:
        sift = cv2.SIFT_create()
        kp, des = sift.compute(img, kp)

        kp1, des1 = sift.compute(patches[0], kp1)
        kp2, des2 = sift.compute(patches[1], kp2)
        kp3, des3 = sift.compute(patches[2], kp3)

    bf = cv2.BFMatcher()

    matches = bf.match(des, des1)
    matches2 = bf.match(des, des2)
    matches3 = bf.match(des, des3)

    matches = sorted(matches, key=lambda val: val.distance)
    matches2 = sorted(matches2, key=lambda val: val.distance)
    matches3= sorted(matches3, key=lambda val: val.distance)

    out = cv2.drawMatches(img, kp, patches[0], kp1, matches[:50], None, flags=2)
    out2 = cv2.drawMatches(img, kp, patches[1], kp2, matches2[:50], None, flags=2)
    out3 = cv2.drawMatches(img, kp, patches[2], kp3, matches3[:50], None, flags=2)

    cv2.imshow('Patch 1', resize_image_ratio(out, 50))
    cv2.imshow('Patch 2', resize_image_ratio(out2, 50))
    cv2.imshow('Patch 3', resize_image_ratio(out3, 50))


def main():

        current_screen = 0

        equalization_create_windows()

        while (True):

            task = cv2.getTrackbarPos('TASK 1-3', 'trackbars')

            if current_screen != task:
                cv2.destroyAllWindows()

                if task == 0:
                    equalization_create_windows()
                    current_screen = 0

                elif task == 1:
                    color_segmentation_create_windows()
                    current_screen = 1

                else:
                    detect_descript_create_windows()
                    current_screen = 2

            if task == 0:
                equalization_task()

            elif task == 1:
                color_segmentation()

            else:
                use_detectors_descriptors()


            k = cv2.waitKey(0) & 0xFF
            if k == 27:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
