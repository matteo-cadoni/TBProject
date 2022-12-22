import cv2 as cv
# add the 2d bounding boxes to the image
def add_bounding_boxes(image, stats):
    """
    Add white rectangles around bacilli, based on conected components

    :param image: image with bacilli to be boxed
    :param coordinates:  coordinates of the center of the bacillus
    """
    for i in range(1, len(stats)):
        x = stats[i][0] - 5
        #x_max = coordinates[i][0]
        y = stats[i][1] - 5
        #y_max = coordinates[i][1]
        h = stats[i][3]
        w = stats[i][2]
        cv.rectangle(image, (x, y), (x + w + 10, y + h + 10), (5000, 255, 255), 1)
    return image