import os
import re
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray


Z_MIN = 0
Z_MAX = 255
Z_RANGE = Z_MAX - Z_MIN + 1


@np.vectorize
def triangle(z):
    return 2 * (z - Z_MIN) / Z_RANGE if z <= (Z_MIN + Z_MAX) / 2 else 2 * (Z_MAX - z) / Z_RANGE


@np.vectorize
def trapezoid(z):
    if z <= Z_MIN + 0.2 * Z_RANGE:
        return 5 * (z - Z_MIN) / Z_RANGE
    if z >= Z_MAX - 0.2 * Z_RANGE:
        return 5 * (Z_MAX - z) / Z_RANGE
    else:
        return 1


def read_images(directory):
    imgs = []
    speeds = []
    shape = None
    for _, _, files in os.walk(directory):
        files = files
        for f in files:
            if os.path.isfile(os.path.join(directory, f)):
                img = cv2.imread(os.path.join(directory, f))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if shape is None:
                        shape = img.shape
                    else:
                        assert shape == img.shape, "different image shape for " + os.path.join(directory, f)
                    temp = re.match('(\d+)_(\d+)\..+', f).groups()
                    speed = int(temp[0]) / int(temp[1])

                    imgs += [img]
                    speeds += [speed]

    indices = np.argsort(speeds)
    return np.array(imgs)[indices], np.array(speeds)[indices]


def sample_points(imgs, num_locations):
    num_imgs, h, w = imgs.shape

    weight = np.mean(imgs, axis=0)
    weight = trapezoid(weight)

    # sample points according to trapezoid brightness
    samples = np.zeros((num_imgs, num_locations), dtype=int)
    linear_idx = np.random.choice(weight.size, size=num_locations, p=weight.ravel() / weight.sum())
    samples_x, samples_y = np.unravel_index(linear_idx, weight.shape)

    for i in range(num_imgs):
        samples[i] = imgs[i, samples_x, samples_y]

    return samples


def solve_log_exposure(samples, speeds, lmbda, weight_func=triangle):
    num_imgs, num_locations = samples.shape
    assert len(speeds) == num_imgs

    # Constraints:
    # num_imgs * num_locations constraints: w(z) * (lG(z) - lE(j) = ln(dt)), pixel value constraint
    # Z_RANGE - 2 constraints:              lmbda * w(z) * (lG(z - 1) - 2 * lG(z) + lG(z + 1) = 0), smooth constraint
    # one constraint:                       lG((Z_MIN + Z_MAX) // 2) = 1, midpoint constraint
    num_constraints = num_imgs * num_locations + (Z_RANGE - 2) + 1
    # Variables:
    # Z_RANGE variables:                    lG(z), log exposure corresponding to pixel value z
    # num_imgs variables:                   lE(j), log irradiance for location i
    num_variables = Z_RANGE + num_locations

    A = np.zeros((num_constraints, num_variables))
    b = np.zeros(num_constraints)

    # adding pixel value constraints
    for i in range(num_imgs):
        for j in range(num_locations):
            k = i * num_locations + j  # iter constraint
            weight = weight_func(samples[i, j])
            A[k, samples[i, j]] = weight
            A[k, Z_RANGE + j] = -weight
            b[k] = weight * np.log(speeds[i])

    # adding smoothness constraint
    for z in range(Z_MIN + 1, Z_MAX):
        k = num_imgs * num_locations + z - 1
        weight = weight_func(z)
        A[k, z - 1] = lmbda * weight
        A[k, z] = -2 * lmbda * weight
        A[k, z + 1] = lmbda * weight
        b[k] = 0

    # adding midpoint constraint
    A[-1, (Z_MIN + Z_MAX) // 2] = 1
    b[-1] = 0

    # solve using SVD
    u, s, v = np.linalg.svd(A, full_matrices=False)
    result = v.T @ np.linalg.inv(np.diag(s)) @ u.T @ b
    return result[:Z_RANGE], result[Z_RANGE:]


def combine_images(imgs, speeds, g, weight_func=triangle):
    g_func = np.vectorize(lambda z: g[z])

    result = np.zeros(imgs[0].shape, dtype=np.float32)
    sum_weights = np.zeros(imgs[0].shape)
    for img, speed in zip(imgs, speeds):
        w = weight_func(img)
        result += (g_func(img) - np.log(speed)) * w
        sum_weights += w

    sum_weights[sum_weights == 0] = 1 / 255.
    result /= sum_weights
    result = (result - np.min(result)) / (np.max(result) - np.min(result))
    result[result == 0] = 1 / 255.

    return result


def tone_mapping(img, method='bilateral'):
    assert len(img.shape) == 3
    assert img.shape[2] == 3
    assert method in ('bilateral', 'log', 'sqrt')

    intensity = rgb2gray(img)
    chrominance = img / intensity[:, :, np.newaxis]

    if method == 'bilateral':
        intensity = np.log(intensity)
        lo_freq = cv2.bilateralFilter(intensity, 3, 64, 64)
        hi_freq = intensity - lo_freq
        lo_freq = (lo_freq - np.max(lo_freq)) * 4 / (np.max(lo_freq) - np.min(lo_freq))
        intensity = np.exp(lo_freq + hi_freq)
    elif method == 'log':
        pass
    elif method == 'sqrt':
        intensity = np.sqrt(np.exp(intensity))

    result = intensity[:, :, np.newaxis] * chrominance
    result **= 1.1
    result = np.clip(result, 0, 1)
    return result


def create_hdr(directory):
    print("creating HDR image for " + directory + '...')
    imgs, speeds = read_images(directory)

    # align images
    aligner = cv2.createAlignMTB(max_bits=2, cut=True)
    aligner.process(imgs, imgs)

    result = np.zeros_like(imgs[0], dtype=np.float32)
    for c in range(3):
        # sample points
        samples = sample_points(imgs[:, :, c], num_locations=500)

        # solve for log exposure curve g
        g, _ = solve_log_exposure(samples, speeds, lmbda=50)

        result[:, :, c] = combine_images(imgs[:, :, :, c], speeds, g)

        plt.imsave(os.path.join('./output', os.path.split(directory)[1] + '_radiance.png'), rgb2gray(result))

    # tone mapping
    method = 'bilateral'
    result = tone_mapping(result, method=method)

    plt.imsave(os.path.join('./output', os.path.split(directory)[1] + '_' + method + '.png'), result)


def main():
    if len(sys.argv) == 1:
        data_dir = './data'
        for _, dirs, _ in os.walk(data_dir):
            for directory in dirs:
                create_hdr(os.path.join(data_dir, directory))
    else:
        for directory in sys.argv[1:]:
            create_hdr(directory)


if __name__ == '__main__':
    main()


