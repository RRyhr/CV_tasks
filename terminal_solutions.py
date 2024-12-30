import cv2
import numpy as np

# 1.  ORB фичи
def find_orb_features(img: np.ndarray) -> np.ndarray:
    orb = cv2.ORB_create()
    keypoints = orb.detect(img, None)
    out_img = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=0) # рисуем ключевые точки
    return out_img # выводит изображение с отрисованными контурами


# поиск SIFT фич
def find_sift_features(img: np.ndarray) -> np.ndarray:
    sift = cv2.SIFT_create()
    keypoints, _ = sift.detectAndCompute(img, None)

    out_img = cv2.drawKeypoints(img, keypoints, None, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return out_img # изображение с отрисованными особенностями

# 3. показывает Canny-границы
def find_canny_edges(img: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(img, 100, 200)
    return edges


# 4. grayscale
def to_grayscale(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


#  5. перевод в HSV
def to_hsv(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # просто другое базисное разбиение изображения
    return hsv


# 6. отразить по правой границе
def flip_right(img: np.ndarray) -> np.ndarray:
    flipped = cv2.flip(img, 1) # flipCode=1 означает отражение по вертикальной оси
    return flipped


# 7. Отразить по нижней границе
def flip_bottom(img: np.ndarray) -> np.ndarray:
    flipped = cv2.flip(img, 0) # flipCode=0 означает отражение по горизонтальной оси
    return flipped


# 8. Поворот на 45 ВОКРУГ ЦЕНТРА
def rotate_45(img: np.ndarray) -> np.ndarray:
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 45, 1.0) # матрица поворота
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

# 9. Поворот на 30 вокруг ЗАДАННОЙ точки
def rotate_30_around_point(img: np.ndarray, point=(100, 100)) -> np.ndarray:
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D(point, 30, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated


# 10. смещение на 10 пикселей вправо
def shift_right_10(img: np.ndarray) -> np.ndarray:
    (h, w) = img.shape[:2]
    M = np.float32([[1, 0, 10], # shift по матрице сдвига: [1 0 tx; 0 1 ty]
                    [0, 1, 0]])
    shifted = cv2.warpAffine(img, M, (w, h))
    return shifted


# 11. яркость
def change_brightness(img: np.ndarray, beta=50) -> np.ndarray:
    # beta > 0 -> более светлое изображение
    # beta < 0 -> более тёмное изображение
    bright = cv2.convertScaleAbs(img, alpha=1.0, beta=beta) # alpha=1 (коэффициент контраста), beta=значение смещения яркости
    return bright


# 12.  контрастность
def change_contrast(img: np.ndarray, alpha=1.5) -> np.ndarray:
    # alpha > 1 -> увеличить контраст
    # alpha < 1 -> уменьшить контраст
    # beta=0 (без смещения яркости)
    contrast = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
    return contrast


# 13. гамма-коррекция
def gamma_correction(img: np.ndarray, gamma=2.0) -> np.ndarray:
    # гамма-коррекция изображения по формуле out = in^(1/gamma).
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8") #таблица LookUpTable(LUT) для преобразования каждого пикселя
    corrected = cv2.LUT(img, table) # Применяем LUT
    return corrected


# 14. гистограммная эквализация
def histogram_equalization(img: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) # преобразуем BGR -> YCrCb
    channels = cv2.split(ycrcb) # разделяем каналы
    cv2.equalizeHist(channels[0], channels[0]) # эквализация по Y channel[0]=Y
    ycrcb_eq = cv2.merge(channels) # склеим каналы обратно
    img_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR) # в BGR
    return img_eq


# 15. тепло
def warm_balance(img: np.ndarray) -> np.ndarray:
    # делает картинку более тёплой – слегка увеличить красный канал или уменьшить синий
    b, g, r = cv2.split(img.astype(np.float32))
    r *= 1.1 # увеличим красный канал
    b *= 0.9 # уменьшим синий канал
    warm_img = cv2.merge([b, g, r]) # обратная сборка изображения
    warm_img = np.clip(warm_img, 0, 255).astype(np.uint8)
    return warm_img


# 16. Холод
def cold_balance(img: np.ndarray) -> np.ndarray:
    # Делает картинку более холодной – слегка увеличить синий канал (или уменьшить красный)
    b, g, r = cv2.split(img.astype(np.float32))
    b *= 1.1 # увеличим синий
    r *= 0.9 # уменьшим красный

    cold_img = cv2.merge([b, g, r])
    cold_img = np.clip(cold_img, 0, 255).astype(np.uint8)
    return cold_img


# 17. изменение палитры по шаблону
def change_color_palette(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET) # меняет цвета местами (по трем размерностям)
    return colored


# 18. бинаризация
def binarize(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # пороговая бинаризация по методу OTSU
    return thresh


# 19. Контур на бинаризованном изображении
def find_contours_on_binary(img: np.ndarray) -> np.ndarray:
    # предполагается, что уже чёрно-белое (бинарное) изображение.

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # если не уверены, что вход бинарный, то переводим
    else:
        gray = img

    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) # трёхканальная копия(BGR) для отрисовки

    cv2.drawContours(out_img, contours, -1, (0, 0, 255), 2)
    return out_img


# 20. контур(фильтр Sobel)
def find_contours_with_filters(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) # Фильтр Собеля
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3) # контур по 3 точка(можно больше, меньше не стоит)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = np.uint8(np.clip(sobel, 0, 255))

    _, sobel_thresh = cv2.threshold(sobel, 50, 255, cv2.THRESH_BINARY)# пороговая бинаризация

    contours, _ = cv2.findContours(sobel_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(out_img, contours, -1, (0, 255, 0), 2)
    return out_img


# 21. Размытие изображения
def blur_image(img: np.ndarray, ksize=5) -> np.ndarray:
    blurred = cv2.blur(img, (ksize, ksize))
    return blurred


# 22. Фильтрация Фурье – (оставить быстрые частоты)
def fourier_high_pass(img: np.ndarray, radius=30) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols, 2), np.float32) # маска (центральная область = 0, остальное = 1)
    cv2.circle(mask, (ccol, crow), radius, (0, 0, 0), -1)

    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift) # обратное смещение и обратное DFT
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX) # нормализация и преобразование к 8-битному
    img_back = img_back.astype(np.uint8)

    return img_back


# 23. Фильтрация Фурье (оставить ТОЛЬКО низкие частоты)
def fourier_low_pass(img: np.ndarray, radius=30) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols, 2), np.float32) # маска
    cv2.circle(mask, (ccol, crow), radius, (1, 1, 1), -1)

    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift) # обратное смещение и обратное DFT
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX) # нормализация и преобразование к 8-битному
    img_back = img_back.astype(np.uint8)
    return img_back


# 24. эрозия
def erode_image(img: np.ndarray, ksize=3, iterations=1) -> np.ndarray: # применяет морфологическую операцию эрозии
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    eroded = cv2.erode(img, kernel, iterations=iterations)
    return eroded


# 25. дилатация
def dilate_image(img: np.ndarray, ksize=3, iterations=1) -> np.ndarray: # морфологическую операцию дилатации

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    dilated = cv2.dilate(img, kernel, iterations=iterations)
    return dilated


#  Операции
OPERATIONS = {
    "orb": find_orb_features,
    "sift": find_sift_features,
    "canny": find_canny_edges,
    "grayscale": to_grayscale,
    "hsv": to_hsv,
    "flip_right": flip_right,
    "flip_bottom": flip_bottom,
    "rotate_45": rotate_45,
    "rotate_30": rotate_30_around_point,
    # функции с доп параметрами:
    "shift_right": shift_right_10,
    "brightness": change_brightness,
    "contrast": change_contrast,
    "gamma": gamma_correction,
    "hist_eq": histogram_equalization,
    "warm": warm_balance,
    "cold": cold_balance,
    "palette": change_color_palette,
    "binarize": binarize,
    "contours_binary": find_contours_on_binary,
    "contours_filters": find_contours_with_filters,
    "blur": blur_image,
    "fourier_high": fourier_high_pass,
    "fourier_low": fourier_low_pass,
    "erode": erode_image,
    "dilate": dilate_image,
}





def print_help():
    help_text = """
Доступные команды и их параметры:

1. orb
   - Описание: Нахождение ORB-фичей.
   - Параметры: нет

2. sift
   - Описание: Нахождение SIFT-фичей.
   - Параметры: нет

3. canny [threshold1] [threshold2]
   - Описание: Поиск границ с использованием Canny.
   - Параметры:
       threshold1 (опционально): Нижний порог (по умолчанию 100)
       threshold2 (опционально): Верхний порог (по умолчанию 200)

4. grayscale
   - Описание: Перевод изображения в градации серого.
   - Параметры: нет

5. hsv
   - Описание: Перевод изображения в цветовое пространство HSV.
   - Параметры: нет

6. flip_right
   - Описание: Отразить изображение по вертикали (по правой границе).
   - Параметры: нет

7. flip_bottom
   - Описание: Отразить изображение по горизонтали (по нижней границе).
   - Параметры: нет

8. rotate_45
   - Описание: Поворот изображения на 45 градусов вокруг центра.
   - Параметры: нет

9. rotate_30 [x] [y]
   - Описание: Поворот изображения на 30 градусов вокруг точки (x, y). По умолчанию (100, 100).
   - Параметры:
       x (опционально): Координата x точки поворота
       y (опционально): Координата y точки поворота

10. shift_right
    - Описание: Смещение изображения на 10 пикселей вправо.
    - Параметры: нет

11. brightness [beta]
    - Описание: Изменение яркости изображения.
    - Параметры:
        beta (опционально): Значение смещения яркости (по умолчанию 50)

12. contrast [alpha]
    - Описание: Изменение контрастности изображения.
    - Параметры:
        alpha (опционально): Коэффициент контрастности (по умолчанию 1.5)

13. gamma [value]
    - Описание: Гамма-коррекция изображения.
    - Параметры:
        value (опционально): Значение гаммы (по умолчанию 2.0)

14. hist_eq
    - Описание: Гистограммная эквализация.
    - Параметры: нет

15. warm
    - Описание: Применение "тёплого" баланса белого.
    - Параметры: нет

16. cold
    - Описание: Применение "холодного" баланса белого.
    - Параметры: нет

17. palette [name]
    - Описание: Изменение цветовой палитры.
    - Параметры:
        name (опционально): Название палитры (по умолчанию 'jet')

18. binarize [method] [threshold]
    - Описание: Бинаризация изображения.
    - Параметры:
        method (опционально): Метод бинаризации ('otsu' или 'fixed', по умолчанию 'otsu')
        threshold (опционально): Порог для метода 'fixed' (по умолчанию 127)

19. contours_binary
    - Описание: Поиск контуров на бинаризованном изображении и их отрисовка.
    - Параметры: нет

20. contours_filters [filter_type] [threshold]
    - Описание: Поиск контуров с использованием фильтров ('sobel' или 'laplacian').
    - Параметры:
        filter_type (опционально): Тип фильтра (по умолчанию 'sobel')
        threshold (опционально): Порог бинаризации (по умолчанию 50)

21. blur [ksize]
    - Описание: Размытие изображения.
    - Параметры:
        ksize (опционально): Размер ядра (по умолчанию 5)

22. fourier_high [radius]
    - Описание: Высокочастотная фильтрация через преобразование Фурье.
    - Параметры:
        radius (опционально): Радиус фильтра (по умолчанию 30)

23. fourier_low [radius]
    - Описание: Низкочастотная фильтрация через преобразование Фурье.
    - Параметры:
        radius (опционально): Радиус фильтра (по умолчанию 30)

24. erode [ksize] [iterations]
    - Описание: Применение эрозии к изображению.
    - Параметры:
        ksize (опционально): Размер ядра (по умолчанию 3)
        iterations (опционально): Количество итераций (по умолчанию 1)

25. dilate [ksize] [iterations]
    - Описание: Применение дилатации к изображению.
    - Параметры:
        ksize (опционально): Размер ядра (по умолчанию 3)
        iterations (опционально): Количество итераций (по умолчанию 1)

Дополнительные команды:
- help: Показать это сообщение помощи.
- save [path]: Сохранить текущее изображение по указанному пути.
- show: Показать текущее изображение.
- reset: Сбросить изображение к исходному.
- exit / quit: Выйти из программы.
"""
    print(help_text)


def main():
    print("=== Интерактивная система обработки изображений ===")
    print("Введите 'help' для списка доступных команд.\n")

    # Считываем путь к изображению
    input_path = input("Введите путь к входному изображению: ").strip()
    img = cv2.imread(input_path)
    if img is None:
        print(f"Не удалось считать изображение по пути: {input_path}")
        return

    original_img = img.copy()
    while True:
        command_input = input("\nВведите команду: ").strip()
        if not command_input:
            continue # пустой ввод

        parts = command_input.split()
        cmd = parts[0].lower()
        params = parts[1:]

        if cmd in ['exit', 'quit']:
            print("Выход из программы.")
            break
        elif cmd == 'help':
            print_help()
            continue
        elif cmd == 'save':
            if len(params) < 1:
                print("Укажите путь для сохранения изображения. Пример: save output.jpg")
                continue
            output_path = params[0]
            success = cv2.imwrite(output_path, img)
            if success:
                print(f"Изображение сохранено по пути: {output_path}")
            else:
                print(f"Не удалось сохранить изображение по пути: {output_path}")
            continue
        elif cmd == 'show':
            cv2.imshow("Текущее изображение", img)
            print("Нажмите любую клавишу в окне изображения, чтобы продолжить...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            continue
        elif cmd == 'reset':
            img = original_img.copy()
            print("Изображение сброшено к исходному.")
            continue
        elif cmd not in OPERATIONS:
            print(f"Неизвестная команда: '{cmd}'. Введите 'help' для списка доступных команд.")
            continue

        operation_func = OPERATIONS[cmd] # функцию операции

        # Операцию с параметрами
        try:
            if cmd == "canny":
                threshold1 = float(params[0]) if len(params) >= 1 else 100
                threshold2 = float(params[1]) if len(params) >= 2 else 200
                result = operation_func(img, threshold1, threshold2)
            elif cmd == "rotate_30":
                if len(params) >= 2:
                    x = int(params[0])
                    y = int(params[1])
                    point = (x, y)
                    result = operation_func(img, point)
                else:
                    result = operation_func(img)
            elif cmd == "brightness":
                beta = float(params[0]) if len(params) >= 1 else 50
                result = operation_func(img, beta)
            elif cmd == "contrast":
                alpha = float(params[0]) if len(params) >= 1 else 1.5
                result = operation_func(img, alpha)
            elif cmd == "gamma":
                gamma = float(params[0]) if len(params) >= 1 else 2.0
                result = operation_func(img, gamma)
            elif cmd == "palette":
                palette = params[0] if len(params) >= 1 else 'jet'
                result = operation_func(img, palette)
            elif cmd == "binarize":
                method = params[0] if len(params) >= 1 else 'otsu'
                threshold = int(params[1]) if len(params) >= 2 else 127
                result = operation_func(img, method, threshold)
            elif cmd == "contours_filters":
                filter_type = params[0] if len(params) >= 1 else 'sobel'
                threshold = int(params[1]) if len(params) >= 2 else 50
                result = operation_func(img, filter_type, threshold)
            elif cmd == "blur":
                ksize = int(params[0]) if len(params) >= 1 else 5
                result = operation_func(img, ksize)
            elif cmd in ["fourier_high", "fourier_low"]:
                radius = int(params[0]) if len(params) >= 1 else 30
                result = operation_func(img, radius)
            elif cmd in ["erode", "dilate"]:
                ksize = int(params[0]) if len(params) >= 1 else 3
                iterations = int(params[1]) if len(params) >= 2 else 1
                result = operation_func(img, ksize, iterations)
            elif cmd == "rotate_30":
                if len(params) >= 2:
                    x = int(params[0])
                    y = int(params[1])
                    point = (x, y)
                    result = operation_func(img, point)
                else:
                    result = operation_func(img)
            else:
                # Операций без параметров
                result = operation_func(img)
            img = result
            print(f"Команда '{cmd}' выполнена успешно.")
        except Exception as e:
            print(f"Ошибка при выполнении команды '{cmd}': {e}")
            continue

    # закрываем все окна
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


