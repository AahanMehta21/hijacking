from PIL import Image
import numpy as np

def letterbox_image(image, size):
    """ Resize image with unchanged aspect ratio using padding.

    Args:
        image: PIL.Image.Image (Jpeg or PNG)
        size: Tuple (416, 416)
    
    Returns:
        new_image: PIL.Image.Image
    """
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    # new_image = Image.new('RGB', size, (0, 0, 0))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image


def image_to_ndarray(image, expand_dims=True):
    """ Convert PIL Image to numpy.ndarray and add batch dimension
    
        Args:
            image: PIL.Image.Image
        
        Returns:
            image_data: numpy.ndarray (1, 416, 416, 3) or (416, 416, 3)

    """
    image_data = np.array(image, dtype='float32')
    image_data /= 255.
    if expand_dims == True:
        image_data = np.expand_dims(image_data, 0)
    if image_data.shape[-1] == 4:
        image_data = image_data[...,0:-1]
    return image_data

def ndarray_to_image(image_data):
    if len(image_data.shape) == 4:
        image_data = np.squeeze(image_data, axis=0)
    image_data = (image_data * 255).astype("uint8")
    return Image.fromarray(image_data)

def load_yolov3_image(img_fpath):
    """ Load and resize an image for yolo3. """
    model_image_size = (416, 416)
    image = Image.open(img_fpath)
    boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image_data

def l1_diff(image1, image2):
    diff = np.abs(image1 - image2)
    return np.sum(diff)

def l0_diff(image1, image2):
    diff = np.abs(image1 - image2)
    return np.count_nonzero(diff)

def l_inf_diff(image1, image2):
    diff = np.abs(image1 - image2)
    return np.max(diff)

if __name__ == "__main__":
    main()