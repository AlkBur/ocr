import cv2
from pdf2image import convert_from_path
import numpy as np
from PIL import Image

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


def prepare_page(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresholded_image = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY)[1]
    inverted_image = cv2.bitwise_not(thresholded_image)

    hor = np.array([[1, 1, 1, 1, 1, 1]])
    vertical_lines_eroded_image = cv2.erode(inverted_image, hor, iterations=10)
    vertical_lines_eroded_image = cv2.dilate(vertical_lines_eroded_image, hor, iterations=10)
    #show_image('vertical_lines_eroded_image', vertical_lines_eroded_image)

    ver = np.array([[1],
                    [1],
                    [1],
                    [1],
                    [1],
                    [1],
                    [1]])
    horizontal_lines_eroded_image = cv2.erode(inverted_image, ver, iterations=10)
    horizontal_lines_eroded_image = cv2.dilate(horizontal_lines_eroded_image, ver, iterations=10)
    #show_image('horizontal_lines_eroded_image', horizontal_lines_eroded_image)

    combined_image = cv2.add(vertical_lines_eroded_image, horizontal_lines_eroded_image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    combined_image_dilated = cv2.dilate(combined_image, kernel, iterations=5)

    image_without_lines = cv2.subtract(inverted_image, combined_image_dilated)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    image_without_lines_noise_removed = cv2.erode(image_without_lines, kernel, iterations=1)
    image_without_lines_noise_removed = cv2.dilate(image_without_lines_noise_removed, kernel, iterations=1)


    # Распознавание текста
    kernel_to_remove_gaps_between_words = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ])
    dilated_image = cv2.dilate(thresholded_image, kernel_to_remove_gaps_between_words, iterations=5)
    simple_kernel = np.ones((5, 5), np.uint8)
    dilated_image = cv2.dilate(dilated_image, simple_kernel, iterations=2)

    contours, hierarchy = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #contours = result[0]
    # The code below is for visualization purposes only.
    # It is not necessary for the OCR to work.
    image_with_contours_drawn = image.copy()
    cv2.drawContours(image_with_contours_drawn, contours, -1, (0, 255, 0), 3)
    show_image('image_with_contours_drawn', image_with_contours_drawn)

    bounding_boxes = []
    image_with_all_bounding_boxes = image.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))
        # This line below is about
        # drawing a rectangle on the image with the shape of
        # the bounding box. Its not needed for the OCR.
        # Its just added for debugging purposes.
        image_with_all_bounding_boxes = cv2.rectangle(image_with_all_bounding_boxes, (x, y), (x + w, y + h),
                                                           (0, 255, 0), 5)

    show_image('image_with_all_bounding_boxes', image_with_all_bounding_boxes)


    return image_without_lines_noise_removed


    # ## Операция предварительной обработки
    # grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # #show_image('gray', grayscale_image)
    #
    # thresholded_image = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #
    # inverted_image = cv2.bitwise_not(thresholded_image)
    # #show_image('inverted_', inverted_image)
    # dilated_image = cv2.dilate(inverted_image, None, iterations=5)
    # #show_image('dilate', dilated_image)
    # contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # image_with_all_contours = img.copy()
    # cv2.drawContours(image_with_all_contours, contours, -1, (0, 255, 0), 3)
    #
    # show_image('contours', image_with_all_contours)
    #
    # rectangular_contours = []
    # for contour in contours:
    #     peri = cv2.arcLength(contour, True)
    #     approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    #     if len(approx) == 4:
    #         rectangular_contours.append(approx)
    #     # Below lines are added to show all rectangular contours
    #     # This is not needed, but it is useful for debugging
    # image_with_only_rectangular_contours = img.copy()
    # cv2.drawContours(image_with_only_rectangular_contours, rectangular_contours, -1, (0, 255, 0), 3)
    #
    #
    # return image_with_only_rectangular_contours

    # img = np.array(page)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # show_image('gray', gray)
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # # show_image('GaussianBlur', gray)
    # edged = cv2.Canny(gray, 75, 200)
    # #return edged
    #
    # # Обнаружение контура
    # cnts, hierancy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # # Найдите контур на изображении
    # # cnts = sorted (cnts, key = cv2.contourArea, reverse = True) [: 5] # Сортировка контуров по размеру области и выбор самого большого контура среди 5 лучших, если есть несколько маленьких билетов
    # print('cnts',len(cnts))
    #
    # for c in cnts:
    #     peri = cv2.arcLength(c, True)  # Окружность, закрытая
    #     # print(peri)
    #     approx=cv2.approxPolyDP(c, 0.02* peri, True)  # Обнаруженный контур может быть дискретными точками, поэтому здесь выполняются приблизительные вычисления, чтобы он сформировал прямоугольник
    #     # Для контроля точности максимальное расстояние от исходного контура до приблизительного контура, если оно меньше, то может быть многоугольником; если больше, то может быть прямоугольным
    #     # True означает закрытый
    #     if len(approx)==4:  # Если обнаружен прямоугольник, разорвать этот абзац, если
    #         screenCnt = approx
    #         print(len(approx))
    #
    # image=img.copy()
    # cv2.drawContours(image, [screenCnt], -1, (0, 0, 255), 2)  # Рисуем контуры, -1 означает рисовать все
    # show_image('contour', image)
    # return image

def run_conversion(input_file):
    print("Starting conversion of %s" % input_file)
    pages = convert_from_path(input_file, 300)
    #img = np.array(convert_images(pages))
    # show_image(img, "start")
    for page in pages:
        img = np.array(page)
        img_rotated = rotate_page(img)
        show_image("rotate", img_rotated)
        img_finaly = prepare_page(img)
        show_image("finaly", img_finaly)


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


