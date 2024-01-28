import math

import cv2
from pdf2image import convert_from_path
import numpy as np
from PIL import Image
import pytesseract

def show_image(text, img):
    cv2.namedWindow(text, cv2.WINDOW_NORMAL)
    cv2.imshow(text, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def merge_images(imgs):
    # for a vertical stacking it is simple: use vstack
    min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
    imgs_comb = np.vstack([i.resize(min_shape) for i in imgs])
    imgs_comb = Image.fromarray(imgs_comb)
    return imgs_comb
    # imgs_comb.save('Trifecta_vertical.jpg')


def rotate_page(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresholded_image = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY)[1]
    inverted_image = cv2.bitwise_not(thresholded_image)

    hor = np.array([[1, 1, 1, 1, 1, 1]])
    vertical_lines_eroded_image = cv2.erode(inverted_image, hor, iterations=10)
    vertical_lines_eroded_image = cv2.dilate(vertical_lines_eroded_image, hor, iterations=10)
    # show_image('vertical_lines_eroded_image', vertical_lines_eroded_image)


    coords = np.column_stack(np.where(vertical_lines_eroded_image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    print(angle)
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    #
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    print(angle)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated

def delete_lines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                              cv2.THRESH_BINARY, 15, -2)

    horizontal = np.copy(bw)
    vertical = np.copy(bw)
    cols = horizontal.shape[1]
    horizontal_size = cols // 30
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    rows = vertical.shape[0]
    verticalsize = rows // 30
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    combined_image = cv2.add(vertical, horizontal)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    combined_image_dilated = cv2.dilate(combined_image, kernel, iterations=5)

    image_without_lines = cv2.subtract(gray, combined_image_dilated)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    image_without_lines_noise_removed = cv2.erode(image_without_lines, kernel, iterations=1)
    image_without_lines_noise_removed = cv2.dilate(image_without_lines_noise_removed, kernel, iterations=1)
    inverted_image = cv2.bitwise_not(image_without_lines_noise_removed)

    noiseless_image = cv2.fastNlMeansDenoising(inverted_image, None, 20, 7, 21)

    return noiseless_image


def scan_upd(image):
    print("Scan UPD file")
    ret, thresh1 = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # Specify structure shape and kernel size.
    # Kernel size increases or decreases the area
    # of the rectangle to be detected.
    # A smaller value like (10, 10) will detect
    # each word instead of a sentence.
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 8))

    # Applying dilation on the threshold image
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=3)

    # Finding contours
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    show_image("upd", image)

    # Creating a copy of image
    im2 = image.copy()

    # A text file is created and flushed
    file = open("/media/sf_OCR/OpenCV/recognized.txt", "w+")
    file.write("")
    file.close()
    # Looping through the identified contours
    # Then rectangular part is cropped and passed on
    # to pytesseract for extracting text from it
    # Extracted text is then written into the text file
    count = 10
    for cnt in contours:
        # rect = cv2.ar(cnt)
        # area = cv2.contourArea(cnt)


        x, y, w, h = cv2.boundingRect(cnt)

        # Drawing a rectangle on copied image
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # # Cropping the text block for giving input to OCR
        # cropped = im2[y:y + h, x:x + w]
        #
        # # Open the file in append mode
        # file = open("recognized.txt", "a")
        #
        # # Apply OCR on the cropped image
        # text = pytesseract.image_to_string(cropped, lang="rus+eng")
        # print("scan: %s" % text)
        #
        # # Appending the text into file
        # file.write(text)
        # file.write("\n")
        #
        # # Close the file
        # file.close()
        #
        # count -= 1
        # if count < 0:
        #     break

    print("Finish scan file")



def run_conversion(input_file):
    print("Starting conversion of %s" % input_file)
    pages = convert_from_path(input_file, 300)

    images = []
    for page in pages:
        img = np.array(page)
        img_rotated = rotate_page(img)
        images.append(Image.fromarray(img_rotated))

    imgs_comb = merge_images(images)
    #imgs_comb.save('/media/sf_OCR/OpenCV/rotate.jpg')
    im_np = np.asarray(imgs_comb)
    img_clear = delete_lines(im_np)
    # show_image("clear", img_clear)

    template_UPD = cv2.imread('/media/sf_OCR/OpenCV/templates/type.png', cv2.IMREAD_GRAYSCALE)
    result = cv2.matchTemplate(template_UPD, img_clear, cv2.TM_CCOEFF)

    loc = np.where(result > 0.5)
    if len(loc[0]) > 0:
        detection = True
    else:
        detection = False

    if detection:
        scan_upd(img_clear)
        # w, h = template_UPD.shape[::-1]
        # # Grab the Max and Min values, plus their locations
        # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        # top_left = max_loc
        # bottom_right = (top_left[0] + w, top_left[1] + h)
        #
        # # Draw the Red Rectangle
        # cv2.rectangle(img_clear, top_left, bottom_right, 0, 2)
    else:
        print("Type file not detection %s" % detection)
        show_image("clear", img_clear)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # input_file = "/media/sf_OCR/OpenCV/26. ООО КРУИЗ _УПД (статус 1) № 26 от 08 апреля 2022 г.pdf"
    # run_conversion(input_file)
    # input_file = "/media/sf_OCR/OpenCV/3955.pdf"
    # run_conversion(input_file)
    # input_file = "/media/sf_OCR/OpenCV/Терес-сервис ООО № М6321 от 27.11.2023 г.pdf"
    # run_conversion(input_file)

    # Угол поворота проверить
    input_file = "/media/sf_OCR/OpenCV/112. УПД № 1760 от 18.04.22.pdf"
    run_conversion(input_file)


