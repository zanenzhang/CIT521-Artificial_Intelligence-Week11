############################################################
# Individual Functions
############################################################

def convolve_greyscale(image, kernel):               # array image of shape (image_height, image_width), numpy array kernel of shape (kernel_height, kernel_width) of floats

    if image is not None:
        imageHeight = len(image)
    else:
        return -1

    if image[0] is not None:
        imageWidth = len(image[0])
    else:
        return -1

    if kernel is not None:
        kernelHeight = len(kernel)
    else:
        return -1

    if kernel[0] is not None:
        kernelWidth = len(kernel[0])
    else:
        return -1

    kernelfliped = np.fliplr(kernel)              #Flip the kernel both ways as instructed
    kernelfliped = np.flipud(kernelfliped)

    horizontalpad = (kernelWidth - 1) // 2
    verticalpad = (kernelHeight - 1) // 2
    paddedImage = np.pad(image, pad_width=[(verticalpad, verticalpad), (horizontalpad, horizontalpad)])     #Pad with zeros as instructed #Padding is [(top row, bottom row),(left col, right col)]
    output = np.empty((imageHeight, imageWidth), dtype=float)    #Output array to store the convolved image

    for x in np.arange(verticalpad, imageHeight + verticalpad):
        for y in np.arange(horizontalpad, imageWidth + horizontalpad):
            region = paddedImage[x - verticalpad: x + verticalpad + 1, y - horizontalpad:y + horizontalpad + 1]
            newValue = float(np.multiply(region, kernelfliped).sum())
            output[x - verticalpad, y - horizontalpad] = newValue

    return output

def convolve_rgb(image, kernel):

    if image is not None:
        channelDepth = len(image)
    else:
        return -1

    if image[0] is not None:
        channelHeight = len(image[0])
    else:
        return -1

    if kernel is not None:
        kernelHeight = len(kernel)
    else:
        return -1

    output = deepcopy(image).astype(float)
    height, width, depth = np.shape(image)
    for i in np.arange(depth):
        imageCopy = image[:, :, i]
        output[:, :, i] = convolve_greyscale(imageCopy, kernel)

    return output


def max_pooling(image, kernel, stride):

    if image is not None:
        imageHeight = len(image)
    else:
        return -1

    if image[0] is not None:
        imageWidth = len(image[0])
    else:
        return -1

    if kernel is not None:
        kernelHeight = kernel[0]
        kernelWidth = kernel[1]
    else:
        return -1

    if stride is not None:
        strideHeight = stride[0]
        strideWidth = stride[1]
    else:
        return -1

    targetHeight = (imageHeight - kernelHeight)//strideHeight + 1
    targetWidth = (imageWidth - kernelWidth)//strideWidth + 1
    output = np.empty((targetHeight, targetWidth))

    i = 0
    j = 0
    for x in np.arange(0, imageHeight - kernelHeight + 1, strideHeight):
        for y in np.arange(0, imageWidth - kernelWidth + 1, strideWidth):
            window = image[x:x+kernelHeight, y:y+kernelWidth]
            localMax = np.amax(window)
            output[i, j] = localMax
            j += 1
        i += 1
        j = 0

    return output


def average_pooling(image, kernel, stride):
    if image is not None:
        imageHeight = len(image)
    else:
        return -1

    if image[0] is not None:
        imageWidth = len(image[0])
    else:
        return -1

    if kernel is not None:
        kernelHeight = kernel[0]
        kernelWidth = kernel[1]
    else:
        return -1

    if stride is not None:
        strideHeight = stride[0]
        strideWidth = stride[1]
    else:
        return -1

    targetHeight = (imageHeight - kernelHeight) // strideHeight + 1
    targetWidth = (imageWidth - kernelWidth) // strideWidth + 1
    output = np.empty((targetHeight, targetWidth))

    i = 0
    j = 0
    for x in np.arange(0, imageHeight - kernelHeight + 1, strideHeight):
        for y in np.arange(0, imageWidth - kernelWidth + 1, strideWidth):
            window = image[x:x + kernelHeight, y:y + kernelWidth]
            localMax = np.mean(window)
            output[i, j] = localMax
            j += 1
        i += 1
        j = 0

    return output

def sigmoid(x):

    output = 1/(1 + np.exp(-x))

    return output


