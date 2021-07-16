import numpy as np
import PIL


def rotate(image, angle):
    return np.array(image.rotate(angle))


def scale(image, factor, h, w):
    new_size = [int(factor * h), int(factor * w)]
    output_image = np.array(image.resize(new_size, PIL.Image.ANTIALIAS))

    if factor < 1:
        output = np.zeros(np.array(image).shape, dtype='int')
        output_center = [int(h / 2), int(w / 2)]
        scaled_shape = output_image.shape

        output[output_center[0] - int(np.floor(scaled_shape[0]/2)): output_center[0] + int(np.ceil(scaled_shape[0]/2)),
               output_center[1] - int(np.floor(scaled_shape[1]/2)): output_center[1] + int(np.ceil(scaled_shape[1]/2)),
               :] = output_image
    else:
        output_center = [int(output_image.shape[0] / 2), int(output_image.shape[1] / 2)]
        scaled_shape = (h, w)
        output = output_image[output_center[0] - int(np.floor(scaled_shape[0] / 2)): output_center[0] +
                              int(np.ceil(scaled_shape[0] / 2)),
                              output_center[1] - int(np.floor(scaled_shape[1] / 2)): output_center[1] +
                              int(np.ceil(scaled_shape[1] / 2)), :]

    return output


def flip(image, axis):
    return np.flip(image, axis=axis)


def noise(image, std):
    image = np.array(image)
    my_noise = np.random.normal(size=image.shape)
    image = ((image / 255.0 + my_noise * std) * 255.0).astype('uint8')
    return np.clip(image, 0, 255)


def transform(image, augmentation):
    h, w, _ = image.shape
    image = PIL.Image.fromarray(image)

    if augmentation == 'rotate':
        angle = np.random.uniform(-10, 10)
        return rotate(image, angle)

    elif augmentation == 'scale':
        factor = np.random.uniform(0.7, 1.4)
        return scale(image, factor, h, w)

    elif augmentation == 'flip':
        return flip(image, 1)

    elif augmentation == 'noise':
        std = np.random.uniform(0.005, 0.05)
        return noise(image, std)


def augment_data(data, labels):
    new_data = data.copy()
    new_labels = labels.copy()

    augmentations = ['rotate', 'flip', 'noise']

    for i in range(len(data)):
        for augmentation in augmentations:
            image = data[i]
            label = labels[i]

            new_data.append(transform(image, augmentation))
            new_labels.append(label)

    return new_data, new_labels
