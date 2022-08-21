#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
from numpy.linalg import det, lstsq, norm
from functools import cmp_to_key
import time
import math
from tqdm import tqdm


float_tolerance = 1e-7
sigma=1.6
num_intervals=3
assumed_blur=0.5
image_border_width=5


array_of_img = []
# Read directory function
def read_directory(directory_name):
    filenumber = len([name for name in os.listdir(directory_name) if os.path.isfile(os.path.join(directory_name, name))])
    for i in range(1,filenumber+1):
        img = cv2.imread(directory_name + "/" + str(i)+".jpg")
        array_of_img.append(img)

def SIFT(image):

    image = image.astype('float32')
    now = time.time()
    base_image = generateBaseImage(image, sigma, assumed_blur) #1
    now = time.time()
    num_octaves = computeNumberOfOctaves(base_image.shape) #2
    print('Compute Number Of Octaves: {}s'.format(time.time()-now))
    now = time.time()
    gaussian_kernels = generateGaussianKernels(sigma, num_intervals) #3
    print('Generate Gaussian Kernels: {}s'.format(time.time()-now))
    now = time.time()
    gaussian_images = generateGaussianImages(base_image, num_octaves, gaussian_kernels) #4
    print('Generate Gaussian Images: {}s'.format(time.time()-now))
    now = time.time()
    dog_images = generateDoGImages(gaussian_images) #5
    print('Generate DoG Images: {}s'.format(time.time()-now))
    now = time.time()
    keypoints = findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width) #6
    print('Find ScaleSpace Extrema: {}s'.format(time.time()-now))
    now = time.time()
    keypoints = removeDuplicateKeypoints(keypoints) #7
    print('Remove Duplicate Keypoints: {}s'.format(time.time()-now))
    now = time.time()
    keypoints = convertKeypointsToInputImageSize(keypoints) #8
    print('Convert Keypoints To Input Image Size: {}s'.format(time.time()-now))
    now = time.time()
    descriptors = generateDescriptors(keypoints, gaussian_images) #9
    print('Generate Descriptors: {}s'.format(time.time()-now))
    
    return keypoints, descriptors     

def generateBaseImage(image, sigma, assumed_blur):
    image = cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    sigma_diff = np.sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
    
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)  # the image blur is now sigma instead of assumed_blur

def computeNumberOfOctaves(image_shape):

    num_of_octaves = int(np.round(np.log(min(image_shape)) / np.log(2) - 2))
    # print('num_of_octaves = {}'.format(num_of_octaves))
    return int(np.round(np.log(min(image_shape)) / np.log(2) - 2))

def generateGaussianKernels(sigma, num_intervals):

    num_images_per_octave = num_intervals + 3
    k = 2 ** (1. / num_intervals)
    gaussian_kernels = np.zeros(num_images_per_octave) 
    gaussian_kernels[0] = sigma

    for image_index in range(1, num_images_per_octave):
        sigma_previous = (k ** (image_index - 1)) * sigma
        sigma_total = k * sigma_previous
        gaussian_kernels[image_index] = np.sqrt(sigma_total ** 2 - sigma_previous ** 2)
    return gaussian_kernels


def generateGaussianImages(image, num_octaves, gaussian_kernels):

    gaussian_images = []
    global img_idx
    for octave_index in range(num_octaves):
        gaussian_images_in_octave = []# first image in octave already has the correct blur
        gaussian_images_in_octave.append(image)
        for gaussian_kernel in gaussian_kernels[1:]:
            image = cv2.GaussianBlur(image, (0, 0), sigmaX=gaussian_kernel, sigmaY=gaussian_kernel)
            gaussian_images_in_octave.append(image)
            if img_idx ==1:
                cv2.imwrite('./output/Image_pyramid/img1/Octave{}_GaussianBlur{}.jpg'.format(octave_index+1,gaussian_kernel), image)
            if img_idx==2:
                cv2.imwrite('./output/Image_pyramid/img2/Octave{}_GaussianBlur{}.jpg'.format(octave_index+1,gaussian_kernel), image)
        gaussian_images.append(gaussian_images_in_octave)
        octave_base = gaussian_images_in_octave[-3]
        image = cv2.resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)), interpolation=cv2.INTER_NEAREST)
        
    return np.array(gaussian_images, dtype=object)

def generateDoGImages(gaussian_images):
    dog_images = []

    for octa,gaussian_images_in_octave in enumerate(gaussian_images):
        dog_images_in_octave = []
        for num,(first_image, second_image) in enumerate(zip(gaussian_images_in_octave, gaussian_images_in_octave[1:])):
            dog_images_in_octave.append(cv2.subtract(second_image, first_image))  
            images = cv2.subtract(second_image, first_image)

            ret,dog = cv2.threshold(images,0,255,cv2.THRESH_BINARY_INV)
            #cv2.imwrite('My Image_oct{}.jpg'.format(octa+1), dog)
            if img_idx==1:
                cv2.imwrite('./output/DOG/img1/Octave{}_GaussianBlur{}.jpg'.format(octa+1,num), dog)
            if img_idx==2:
                cv2.imwrite('./output/DOG/img2/Octave{}_GaussianBlur{}.jpg'.format(octa+1,num), dog)
        dog_images.append(dog_images_in_octave)
    return np.array(dog_images, dtype=object)


def findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width, contrast_threshold=0.04):

    threshold = np.floor(0.5 * contrast_threshold / num_intervals * 255)  # from OpenCV implementation
    keypoints = []

    for octave_index, dog_images_in_octave in enumerate(dog_images):
        for image_index, (first_image, second_image, third_image) in enumerate(zip(dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:])):
            # (i, j) is the center of the 3x3 array
            for i in range(image_border_width, first_image.shape[0] - image_border_width):
                for j in range(image_border_width, first_image.shape[1] - image_border_width):
                    if isPixelAnExtremum(first_image[i-1:i+2, j-1:j+2], second_image[i-1:i+2, j-1:j+2], third_image[i-1:i+2, j-1:j+2], threshold):
                        localization_result = localizeExtremumViaQuadraticFit(i, j, image_index + 1, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width)
                        if localization_result is not None:
                            keypoint, localized_image_index = localization_result
                            keypoints_with_orientations = computeKeypointsWithOrientations(keypoint, octave_index, gaussian_images[octave_index][localized_image_index])
                            for keypoint_with_orientation in keypoints_with_orientations:
                                keypoints.append(keypoint_with_orientation)
    return keypoints

def isPixelAnExtremum(first_subimage, second_subimage, third_subimage, threshold):

    center_pixel_value = second_subimage[1, 1]
    if abs(center_pixel_value) > threshold:
        if center_pixel_value > 0:
            return np.all(center_pixel_value >= first_subimage) and np.all(center_pixel_value >= third_subimage) and np.all(center_pixel_value >= second_subimage[0, :]) and np.all(center_pixel_value >= second_subimage[2, :]) and center_pixel_value >= second_subimage[1, 0] and center_pixel_value >= second_subimage[1, 2]
        elif center_pixel_value < 0:
            return np.all(center_pixel_value <= first_subimage) and np.all(center_pixel_value <= third_subimage) and np.all(center_pixel_value <= second_subimage[0, :]) and np.all(center_pixel_value <= second_subimage[2, :]) and center_pixel_value <= second_subimage[1, 0] and center_pixel_value <= second_subimage[1, 2]
    return False

def localizeExtremumViaQuadraticFit(i, j, image_index, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width, eigenvalue_ratio=10, num_attempts_until_convergence=5):
    extremum_is_outside_image = False
    image_shape = dog_images_in_octave[0].shape
    for attempt_index in range(num_attempts_until_convergence):
        
        first_image, second_image, third_image = dog_images_in_octave[image_index-1:image_index+2]
        pixel_cube = np.stack([first_image[i-1:i+2, j-1:j+2],
                            second_image[i-1:i+2, j-1:j+2],
                            third_image[i-1:i+2, j-1:j+2]]).astype('float32') / 255.
        gradient = computeGradientAtCenterPixel(pixel_cube)
        hessian = computeHessianAtCenterPixel(pixel_cube)
        extremum_update = -lstsq(hessian, gradient, rcond=None)[0]
        if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2]) < 0.5:
            break
        j += int(np.round(extremum_update[0]))
        i += int(np.round(extremum_update[1]))
        image_index += int(np.round(extremum_update[2]))
        if i < image_border_width or i >= image_shape[0] - image_border_width or j < image_border_width or j >= image_shape[1] - image_border_width or image_index < 1 or image_index > num_intervals:
            extremum_is_outside_image = True
            break
    if extremum_is_outside_image:
        return None
    if attempt_index >= num_attempts_until_convergence - 1:
        return None
    functionValueAtUpdatedExtremum = pixel_cube[1, 1, 1] + 0.5 * np.dot(gradient, extremum_update)
    if abs(functionValueAtUpdatedExtremum) * num_intervals >= contrast_threshold:
        xy_hessian = hessian[:2, :2]
        xy_hessian_trace = np.trace(xy_hessian)
        xy_hessian_det = det(xy_hessian)
        if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
            keypoint = cv2.KeyPoint()
            keypoint.pt = ((j + extremum_update[0]) * (2 ** octave_index), (i + extremum_update[1]) * (2 ** octave_index))
            keypoint.octave = octave_index + image_index * (2 ** 8) + int(np.round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
            keypoint.size = sigma * (2 ** ((image_index + extremum_update[2]) / np.float32(num_intervals))) * (2 ** (octave_index + 1))  
            keypoint.response = abs(functionValueAtUpdatedExtremum)
            return keypoint, image_index
    return None

def computeGradientAtCenterPixel(pixel_array):
    dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
    dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
    ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
    return np.array([dx, dy, ds])

def computeHessianAtCenterPixel(pixel_array):
    center_pixel_value = pixel_array[1, 1, 1]
    dxx = pixel_array[1, 1, 2] - 2 * center_pixel_value + pixel_array[1, 1, 0]
    dyy = pixel_array[1, 2, 1] - 2 * center_pixel_value + pixel_array[1, 0, 1]
    dss = pixel_array[2, 1, 1] - 2 * center_pixel_value + pixel_array[0, 1, 1]
    dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] - pixel_array[1, 0, 2] + pixel_array[1, 0, 0])
    dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
    dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
    return np.array([[dxx, dxy, dxs], 
                  [dxy, dyy, dys],
                  [dxs, dys, dss]])

def computeKeypointsWithOrientations(keypoint, octave_index, gaussian_image, radius_factor=3, num_bins=36, peak_ratio=0.8, scale_factor=1.5):

    keypoints_with_orientations = []
    image_shape = gaussian_image.shape

    scale = scale_factor * keypoint.size / np.float32(2 ** (octave_index + 1))  # compare with keypoint.size computation in localizeExtremumViaQuadraticFit()
    radius = int(np.round(radius_factor * scale))
    weight_factor = -0.5 / (scale ** 2)
    raw_histogram = np.zeros(num_bins)
    smooth_histogram = np.zeros(num_bins)

    for i in range(-radius, radius + 1):
        region_y = int(np.round(keypoint.pt[1] / np.float32(2 ** octave_index))) + i
        if region_y > 0 and region_y < image_shape[0] - 1:
            for j in range(-radius, radius + 1):
                region_x = int(np.round(keypoint.pt[0] / np.float32(2 ** octave_index))) + j
                if region_x > 0 and region_x < image_shape[1] - 1:
                    dx = gaussian_image[region_y, region_x + 1] - gaussian_image[region_y, region_x - 1]
                    dy = gaussian_image[region_y - 1, region_x] - gaussian_image[region_y + 1, region_x]
                    gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                    gradient_orientation = np.rad2deg(np.arctan2(dy, dx))
                    weight = np.exp(weight_factor * (i ** 2 + j ** 2))  # constant in front of exponential can be dropped because we will find peaks later
                    histogram_index = int(np.round(gradient_orientation * num_bins / 360.))
                    raw_histogram[histogram_index % num_bins] += weight * gradient_magnitude
    for n in range(num_bins):
        smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) + raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.
    orientation_max = max(smooth_histogram)
    orientation_peaks = np.where(np.logical_and(smooth_histogram > np.roll(smooth_histogram, 1), smooth_histogram > np.roll(smooth_histogram, -1)))[0]
    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]
        if peak_value >= peak_ratio * orientation_max:
            # Quadratic peak interpolation
            left_value = smooth_histogram[(peak_index - 1) % num_bins]
            right_value = smooth_histogram[(peak_index + 1) % num_bins]
            interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % num_bins
            orientation = 360. - interpolated_peak_index * 360. / num_bins
            if abs(orientation - 360.) < float_tolerance:
                orientation = 0
            new_keypoint = cv2.KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
            keypoints_with_orientations.append(new_keypoint)
    return keypoints_with_orientations

def compareKeypoints(keypoint1, keypoint2):
    """Return True if keypoint1 is less than keypoint2
    """
    if keypoint1.pt[0] != keypoint2.pt[0]:
        return keypoint1.pt[0] - keypoint2.pt[0]
    if keypoint1.pt[1] != keypoint2.pt[1]:
        return keypoint1.pt[1] - keypoint2.pt[1]
    if keypoint1.size != keypoint2.size:
        return keypoint2.size - keypoint1.size
    if keypoint1.angle != keypoint2.angle:
        return keypoint1.angle - keypoint2.angle
    if keypoint1.response != keypoint2.response:
        return keypoint2.response - keypoint1.response
    if keypoint1.octave != keypoint2.octave:
        return keypoint2.octave - keypoint1.octave
    return keypoint2.class_id - keypoint1.class_id

def removeDuplicateKeypoints(keypoints):
    if len(keypoints) < 2:
        return keypoints

    keypoints.sort(key=cmp_to_key(compareKeypoints))
    unique_keypoints = [keypoints[0]]

    for next_keypoint in keypoints[1:]:
        last_unique_keypoint = unique_keypoints[-1]
        if last_unique_keypoint.pt[0] != next_keypoint.pt[0] or            last_unique_keypoint.pt[1] != next_keypoint.pt[1] or            last_unique_keypoint.size != next_keypoint.size or            last_unique_keypoint.angle != next_keypoint.angle:
            unique_keypoints.append(next_keypoint)
    return unique_keypoints

def convertKeypointsToInputImageSize(keypoints):
    """Convert keypoint point, size, and octave to input image size
    """
    converted_keypoints = []
    for keypoint in keypoints:
        keypoint.pt = tuple(0.5 * np.array(keypoint.pt))
        keypoint.size *= 0.5
        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
        converted_keypoints.append(keypoint)
    return converted_keypoints

def unpackOctave(keypoint):
    """Compute octave, layer, and scale from a keypoint
    """
    octave = keypoint.octave & 255
    layer = (keypoint.octave >> 8) & 255
    if octave >= 128:
        octave = octave | -128
    scale = 1 / np.float32(1 << octave) if octave >= 0 else np.float32(1 << -octave)
    return octave, layer, scale

def generateDescriptors(keypoints, gaussian_images, window_width=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):

    descriptors = []

    for keypoint in keypoints:
        octave, layer, scale = unpackOctave(keypoint)
        gaussian_image = gaussian_images[octave + 1, layer]
        num_rows, num_cols = gaussian_image.shape
        point = np.round(scale * np.array(keypoint.pt)).astype('int')
        bins_per_degree = num_bins / 360.
        angle = 360. - keypoint.angle
        cos_angle = np.cos(np.deg2rad(angle))
        sin_angle = np.sin(np.deg2rad(angle))
        weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
        row_bin_list = []
        col_bin_list = []
        magnitude_list = []
        orientation_bin_list = []
        histogram_tensor = np.zeros((window_width + 2, window_width + 2, num_bins))   # first two dimensions are increased by 2 to account for border effects

        # Descriptor window size (described by half_width) follows OpenCV convention
        hist_width = scale_multiplier * 0.5 * scale * keypoint.size
        half_width = int(np.round(hist_width * np.sqrt(2) * (window_width + 1) * 0.5))   # sqrt(2) corresponds to diagonal length of a pixel
        half_width = int(min(half_width, np.sqrt(num_rows ** 2 + num_cols ** 2)))     # ensure half_width lies within image

        for row in range(-half_width, half_width + 1):
            for col in range(-half_width, half_width + 1):
                row_rot = col * sin_angle + row * cos_angle
                col_rot = col * cos_angle - row * sin_angle
                row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
                if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
                    window_row = int(np.round(point[1] + row))
                    window_col = int(np.round(point[0] + col))
                    if window_row > 0 and window_row < num_rows - 1 and window_col > 0 and window_col < num_cols - 1:
                        dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                        dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                        gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                        gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                        weight = np.exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                        row_bin_list.append(row_bin)
                        col_bin_list.append(col_bin)
                        magnitude_list.append(weight * gradient_magnitude)
                        orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)

        for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
            row_bin_floor, col_bin_floor, orientation_bin_floor = np.floor([row_bin, col_bin, orientation_bin]).astype(int)
            row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
            if orientation_bin_floor < 0:
                orientation_bin_floor += num_bins
            if orientation_bin_floor >= num_bins:
                orientation_bin_floor -= num_bins

            c1 = magnitude * row_fraction
            c0 = magnitude * (1 - row_fraction)
            c11 = c1 * col_fraction
            c10 = c1 * (1 - col_fraction)
            c01 = c0 * col_fraction
            c00 = c0 * (1 - col_fraction)
            c111 = c11 * orientation_fraction
            c110 = c11 * (1 - orientation_fraction)
            c101 = c10 * orientation_fraction
            c100 = c10 * (1 - orientation_fraction)
            c011 = c01 * orientation_fraction
            c010 = c01 * (1 - orientation_fraction)
            c001 = c00 * orientation_fraction
            c000 = c00 * (1 - orientation_fraction)

            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

        descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()  # Remove histogram borders
        threshold = norm(descriptor_vector) * descriptor_max_value
        descriptor_vector[descriptor_vector > threshold] = threshold
        descriptor_vector /= max(norm(descriptor_vector), float_tolerance)
        descriptor_vector = np.round(512 * descriptor_vector)
        descriptor_vector[descriptor_vector < 0] = 0
        descriptor_vector[descriptor_vector > 255] = 255
        descriptors.append(descriptor_vector)
    return np.array(descriptors, dtype='float32')

def get_images_path(path):
    for root, dirs, files in os.walk(dirname):
        pass
    return files

def removeBlackBorder(img):
        '''
        Remove img's the black border 
        '''
        h, w = img.shape[:2]
        reduced_h, reduced_w = h, w
        # right to left
        for col in range(w - 1, -1, -1):
            all_black = True
            for i in range(h):
                if (np.count_nonzero(img[i, col]) > 0):
                    all_black = False
                    break
            if (all_black == True):
                reduced_w = reduced_w - 1
                
        # bottom to top 
        for row in range(h - 1, -1, -1):
            all_black = True
            for i in range(reduced_w):
                if (np.count_nonzero(img[row, i]) > 0):
                    all_black = False
                    break
            if (all_black == True):
                reduced_h = reduced_h - 1
        
        return img[:reduced_h, :reduced_w]

def warp(imgs, HomoMat, blending_mode,HomoMat_mode):
    '''
       Warp image to create panoramic image
       There are three different blending method - noBlending、linearBlending、linearBlendingWithConstant
    '''


    img_left, img_right = imgs
    (hl, wl) = img_left.shape[:2]
    (hr, wr) = img_right.shape[:2]

    stitch_img = np.zeros( (max(hl, hr), wl + wr, 3), dtype="int") # create the (stitch)big image accroding the imgs height and width 

    if (blending_mode == "noBlending"):
        stitch_img[:hl, :wl] = img_left
        # print("noBlending")

    for i in range(stitch_img.shape[0]):
        for j in range(stitch_img.shape[1]):
            coor = np.array([j, i, 1])
            
            img_right_coor = HomoMat @ coor
            img_right_coor /= img_right_coor[2]

            # you can try like nearest neighbors or interpolation  
            y, x = int(round(img_right_coor[0])), int(round(img_right_coor[1])) # y for width, x for height

            # if the computed coordination not in the (hegiht, width) of right image, it's not need to be process 
            if (x < 0 or x >= hr or y < 0 or y >= wr):
                continue
            # else we need the tranform for this pixel
            stitch_img[i, j] = img_right[x, y]
            
    # remove the black border
    stitch_img = removeBlackBorder(stitch_img)
    return stitch_img

def cylindricalWarp(img, K):
    """This function returns the cylindrical warp for a given image and intrinsics matrix K"""
    h_,w_ = img.shape[:2]
    # pixel coordinates
    y_i, x_i = np.indices((h_,w_))
    X = np.stack([x_i,y_i,np.ones_like(x_i)],axis=-1).reshape(h_*w_,3) # to homog
    Kinv = np.linalg.inv(K) 
    X = Kinv.dot(X.T).T # normalized coords
    # calculate cylindrical coords (sin\theta, h, cos\theta)
    A = np.stack([np.sin(X[:,0]),X[:,1],np.cos(X[:,0])],axis=-1).reshape(w_*h_,3)
    B = K.dot(A.T).T # project back to image-pixels plane
    # back from homog coords
    B = B[:,:-1] / B[:,[-1]]
    # make sure warp coords only within image bounds
    B[(B[:,0] < 0) | (B[:,0] >= w_) | (B[:,1] < 0) | (B[:,1] >= h_)] = -1
    B = B.reshape(h_,w_,-1)
    
    #img_rgba = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA) # for transparent borders...
    img_rgba = img
    # print(img_rgba.shape)
    # warp the image according to cylindrical coords
    return cv2.remap(img_rgba, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv2.INTER_AREA)#, borderMode=cv2.BORDER_TRANSPARENT)





def matching(img1_color,img2_color,kp1,kp2,des1,des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append([m])
    img_match = cv2.drawMatchesKnn(img1_color, kp1, img2_color, kp2, good[:20], None, flags=2)
    return img_match,good

def find_H(kp1,kp2,good):
    MIN_MATCH_COUNT=20
    src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)  #Get keypoints corrdinary
    dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

    return H

import time 
start_time = time.time()
with open('testfile.txt', 'r') as f:
    dirnames = f.readlines()
for dirname in dirnames:
    dirname = dirname.strip()
    if dirname=='NCTU':
        # print('Next')
        continue

    num_img = len([name for name in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, name))])
    print('Number of Image: {}'.format(num_img))
    #-------
    #num_img=5
    #dirname = 'parrington'
    #--------
    first_img = math.ceil(num_img/2.0)
    for iteration in tqdm(range(math.floor(num_img/2))):

        if iteration==0:
            img1 = cv2.imread(os.path.join(dirname,str(first_img)+'.jpg'), 0)   # zero for grayscaleprint
            img2 = cv2.imread(os.path.join(dirname,str(first_img+1)+'.jpg'), 0)  
            img3 = cv2.imread(os.path.join(dirname,str(first_img-1)+'.jpg'), 0) 
            img4 = img1
            img1_color = cv2.imread(os.path.join(dirname,str(first_img)+'.jpg'))
            img2_color = cv2.imread(os.path.join(dirname,str(first_img+1)+'.jpg'))
            img3_color = cv2.imread(os.path.join(dirname,str(first_img-1)+'.jpg'))
            img4_color = img1_color
        else:
            img1 = img2
            img2 = cv2.imread(os.path.join(dirname,str(first_img+iteration+1)+'.jpg'), 0)
            img4 = img3
            img3 = cv2.imread(os.path.join(dirname,str(first_img-iteration-1)+'.jpg'), 0)   
            img1_color = img2_color
            img2_color = cv2.imread(os.path.join(dirname,str(first_img+iteration+1)+'.jpg')) 
            img4_color = img3_color
            img3_color = cv2.imread(os.path.join(dirname,str(first_img-iteration-1)+'.jpg')) 

        print("Image Shape: {}".format(img1.shape))

        MIN_MATCH_COUNT = 20
        h, w = img1.shape[:2]
        K = np.array([[800,0,w/2],[0,800,h/2],[0,0,1]]) # mock intrinsics
        img1 = cylindricalWarp(img1, K)
        img2 = cylindricalWarp(img2, K)
        img3 = cylindricalWarp(img3, K)
        img4 = cylindricalWarp(img4, K)
        # img1_color = cylindricalWarp(img1_color, K)
        # img2_color = cylindricalWarp(img2_color, K)
        # img3_color = cylindricalWarp(img3_color, K)
        # img4_color = cylindricalWarp(img4_color, K)
        # cv2.imwrite(os.path.join(dirname,'cylindricalWarp.jpg'), img1)
        
        print('============== Start sift ==============')
        if iteration==0:   
            img_idx = 1
            kp1, des1 = SIFT(img1)
            img_idx = 2
            kp2, des2 = SIFT(img2)
            img_idx = 2
            kp3, des3 = SIFT(img3)
            kp4,des4 = kp1, des1 
        else:
            kp1, des1 = kp2, des2
            kp2, des2 = SIFT(img2)
            kp4,des4 = kp3, des3
            kp3, des3 = SIFT(img3)

        print('============== Finish sift ==============')
        
        # =========== #
        #   MATCHING  #
        # =========== #
        img_match_1,good_1 = matching(img1,img2,kp1,kp2,des1,des2)
        img_match_2,good_2 = matching(img3,img4,kp3,kp4,des3,des4)
        # cv2.imwrite(os.path.join("/data3/DS_HW/HW3-2/hw3/hw3-1/8_9_save",'tmp1.jpg'), img_match_1)
        # cv2.imwrite(os.path.join("/data3/DS_HW/HW3-2/hw3/hw3-1/8_9_save",'tmp2.jpg'), img_match_2)

        print('Finish_matching')

        # MIN_MATCH_COUNT = 20
        # h, w = img1.shape[:2]
        # K = np.array([[800,0,w/2],[0,800,h/2],[0,0,1]]) # mock intrinsics
        # img_cyl_1 = cylindricalWarp(img1_color, K)
        # # cv2.imwrite('img_cyl_1.jpg', img_cyl_1)
        # img_cyl_2 = cylindricalWarp(img2_color, K)          # Try img Cylindrical Projection
        # img_cyl_3 = cylindricalWarp(img3_color, K)       
        # img_cyl_4 = cylindricalWarp(img4_color, K)      


        img_cyl_1 = img1_color
        img_cyl_2 = img2_color 
        img_cyl_3 = img3_color 
        img_cyl_4 = img4_color
        if iteration>0:
            img_cyl_1 = result_2.copy()
        if iteration>0:
           last_H_1 = H_1.copy()
           last_H_2 = H_2.copy()


        H_1 = find_H(kp1,kp2,good_1)
        H_2 = find_H(kp3,kp4,good_2)

        if iteration>0:
            H_1 = H_1.dot(last_H_1.dot(last_H_2))
            H_2 =H_2 
        result_1 = warp([img_cyl_1, img_cyl_2],H_1,"noBlending",'l2r')
        result_2 = warp([img_cyl_3, result_1],H_2,"noBlending",'r2l')

        cv2.imwrite(os.path.join("./output/image_stitching_results",'iteration_{}_1.jpg'.format(iteration+1)), result_1)
        cv2.imwrite(os.path.join("./output/image_stitching_results",'iteration_{}_2.jpg'.format(iteration+1)), result_2)
        cv2.imwrite(os.path.join("./output/image_stitching_results",'results.jpg'.format(iteration+1)), result_2)
        last_H_1 = H_1.copy()
        last_H_2 = H_2.copy()        
        print('Finish')
end_time = time.time()
print('Use: {}s'.format(abs(start_time-end_time)))


