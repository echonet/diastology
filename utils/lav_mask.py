import numpy as np
import skimage
from scipy.interpolate import splev, splprep
from skimage import filters, measure
from skimage.restoration import denoise_bilateral
from sklearn.linear_model import LinearRegression

H_PIX = 0.038512102720240304
W_PIX = 0.038512102720240304

def fuzzy_equals(a,b,thresh=0.25):
    if abs(a-b)<=thresh: 
        return True 
    else: 
        return False
    
def filter_areas(area,min_area=400):
    area = [a for a in area if a>min_area]
    min_idx = int(len(area)*0.05)
    max_idx = int(len(area)*0.95)
    sorted_area = sorted(area)
    cleaned_area = sorted_area[min_idx:max_idx]
    return cleaned_area

def get_la_length(pm,pend,n_discs=20):
    la_length = ((pend[0]-pm[0])**2+(pend[1]-pm[1])**2)**0.5
    h = la_length/n_discs
    return la_length,h

def get_la_vals(mask,n_discs=20): 
    _,contour,p1,p2,pend,*_ = process_mask_to_points(mask)
    pm = (p1+p2)/2
    length,h = get_la_length(pm,pend)
    m_mitral = (p2[1]-p1[1]) / (p2[0]-p1[0])
    m_length = (pend[1]-pm[1]) / (pend[0]-pm[0])
    return contour,m_mitral,m_length,pm,length,h

def get_intersection(contour,m,b,thresh=0.25):
    endpts = []
    for c in contour: 
        if len(endpts)==2: 
            return endpts
        x = c[0]
        y = c[1]
        if len(endpts)>0: 
            endpt_x = endpts[0][0]
            if fuzzy_equals(x,endpt_x,2):
                continue 
        y_parallel = m*x+b
        if fuzzy_equals(y,y_parallel,thresh):
            endpts.append([x,y])
    return endpts

def find_perpendicular(contour,m_mitral,pm):
    if m_mitral==0: 
        m_mitral = 0.01
    m_perpend = -1/m_mitral
    b_perpend = pm[1]-(m_perpend*pm[0])
    la_end = get_intersection(contour,m_perpend,b_perpend,5.)
    # Return slope, y-intercept, and [x,y] coordinates of endpoint
    return m_perpend,b_perpend,la_end[0]

def find_axes(contour,m_mitral,m_perpend,pm,la_end,la_end2=None,n_discs=21):
    axes = []
    endpts = []
    # Get LA length + height of discs based on num_discs
    # length1 = ((la_end[1]-pm[1])**2 + (la_end[0]-pm[0])**2)**0.5 
    length1 = np.linalg.norm(la_end-pm)
    la_length = length1
    if la_end2: 
        # length2 = ((la_end2[1]-pm[1])**2 + (la_end2[0]-pm[0])**2)**0.5 
        length2 = np.linalg.norm(la_end2-pm)
        if length1 > length2: 
            la_length = length2 
    h = la_length/n_discs
    for i in range(1,n_discs):
        ##   get disc center: 
        disc_y = pm[1] + h * i # Traverse line from midpoint of mitral plane to end of LA by disc height h 
        disc_x = (h*i / m_perpend) + pm[0] # Solve for x using line equation dy = m*(x_1 - x_0)
        if disc_y > la_end[1]: 
            print('Surpassed left atrial length')
            break
        ## get b for line that 1) passes through disc center 2) parallel to mitral plane
        b = disc_y - m_mitral*disc_x 
        ## get intersecting contour points 
        disc_endpts = get_intersection(contour,m_mitral,b,0.25)
        disc_1 = np.array(disc_endpts[0])
        disc_2 = np.array(disc_endpts[1])
        ## get distance between disc endpoints 
        # length_axis = ((disc_2[1]-disc_1[1])**2 + (disc_2[0]-disc_1[0])**2)**0.5 
        length_axis = np.linalg.norm(disc_2-disc_1)
        axes.append(length_axis)
        endpts.append([disc_1,disc_2])
    return h,la_length,np.array(axes),endpts

def calc_mod_volume(h,a4c_axes,a2c_axes=None): 
    h_cm = h*H_PIX
    print('Disc height (cm): ',h,h_cm)
    a4c_axes_cm = W_PIX*a4c_axes
    if a2c_axes is not None: 
        a2c_axes_cm = W_PIX*a2c_axes
        biplane_axes = list(zip(a4c_axes_cm,a2c_axes_cm))
        volume = np.pi/4.*sum([a[0]*a[1]*h_cm for a in biplane_axes])
    else: 
        volume = sum([np.pi*(a/2.)**2*h_cm for a in a4c_axes_cm])
    return volume

###----- FUNKCJE POMOCNICZE
def check_and_shift_edge(points, p1, p2):
    """
    Sprawdza i przesuwa krawędź pomiędzy dwoma punktami w konturze, tak aby punkty były uporządkowane.
    Jeżeli punkty P1 i P2 znajdują się na końcach konturu, nie dokonuje żadnych zmian.

    Parameters:
        points (np.ndarray): Zbiór punktów konturu.
        p1 (np.ndarray): Punkt P1 na konturze.
        p2 (np.ndarray): Punkt P2 na konturze.

    Returns:
        np.ndarray: Zaktualizowany zbiór punktów konturu.
    """
    # Znalezienie indeksów punktów P1 i P2
    idx_p1 = np.where(np.all(points == p1, axis=1))[0][0]
    idx_p2 = np.where(np.all(points == p2, axis=1))[0][0]

    min_idx = np.min([idx_p1, idx_p2])
    max_idx = np.max([idx_p1, idx_p2])

    # Sprawdzamy, który punkt ma mniejszy indeks
    if (max_idx > 1) & (min_idx == 0):
        points_new = points
    else:
        points_new = np.roll(
            points, -(min_idx + 1), axis=0
        )  # Przesuwamy punkty tak, aby P1 i P2 były w odpowiedniej kolejności

    return points_new


def find_mitral_plane(points):
    """
    Znajduje punkty tworzące mitral plane (najdłuższą linię w wielokącie).
    Mitral plane to linia łącząca dwa punkty konturu o największej odległości.

    Parameters:
        points (np.ndarray): Zbiór punktów tworzących kontur wielokąta.

    Returns:
        tuple:
            - P1 (np.ndarray): Pierwszy punkt mitral plane.
            - P2 (np.ndarray): Drugi punkt mitral plane.
            - mitral_idx (int): Indeks punktu o największej odległości.
            - max_distance (float): Największa odległość pomiędzy punktami.
    """
    # Obliczanie odległości pomiędzy kolejnymi punktami
    distances = np.sqrt(np.sum(np.diff(points, axis=0, append=points[:1]) ** 2, axis=1))

    # Indeks punktu o największej odległości
    mitral_idx = np.argmax(distances)
    P1, P2 = points[mitral_idx], points[(mitral_idx + 1) % len(points)]

    # Największa odległość
    max_distance = distances[mitral_idx]

    return P1, P2, mitral_idx, max_distance


def smooth_polygon(points, smoothness=0.5, num_points=1000):
    """
    Wygładza wielokąt za pomocą splina, nie usuwając żadnych punktów.
    Tworzy krzywą, która przechodzi przez wszystkie punkty wielokąta.

    Parameters:
        points (np.ndarray): Punkty tworzące kontur wielokąta.
        smoothness (float): Stopień wygładzenia (parametr s dla splprep).
        num_points (int): Liczba wygenerowanych punktów na krzywej.

    Returns:
        np.ndarray: Wygładzone punkty na krzywej.
    """
    # Tworzymy splina dla wszystkich punktów
    tck, _ = splprep([points[:, 0], points[:, 1]], s=smoothness)

    # Generowanie punktów na krzywej splina
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck)

    # Łączenie wygładzonych punktów w jeden zbiór
    smooth_points = np.column_stack([x_new, y_new])

    return smooth_points


def point_of_bottom(points):
    """
    Znajduje punkt konturu o największej wartości współrzędnej 'y'.

    Parameters:
        points (np.ndarray): Zbiór punktów konturu w formacie (x, y).

    Returns:
        np.ndarray: Punkt konturu o największej wartości 'y'.
    """
    # Znalezienie punktu o największej wartości 'y'
    index_of_bottom = np.argmax(points[:, 1])
    bottom_point = points[index_of_bottom]

    return bottom_point


def rasterize_polygon(smooth_points, img_shape):
    """
    Rasteryzacja wielokąta: generowanie maski binarnej na podstawie współrzędnych punktów konturu.

    Parameters:
        smooth_points (np.ndarray): Współrzędne punktów konturu.
        img_shape (tuple): Wymiary obrazu (wysokość, szerokość).

    Returns:
        np.ndarray: Maska binarna przedstawiająca rasteryzowany wielokąt.
    """
    # Generowanie indeksów punktów w obrazie
    r, c = skimage.draw.polygon(
        np.rint(smooth_points[:, 1]).astype(np.int_),
        np.rint(smooth_points[:, 0]).astype(np.int_),
        img_shape,
    )

    # Tworzenie maski binarnej
    mask = np.zeros(img_shape, np.float32)
    mask[r, c] = 1

    return mask


def vector_to_bitmap(example, mask_size, smooth=True):
    """
    Przekształca wektor punktów do bitmapy. Działa na bazie kilku etapów:
    wyznaczania mitral plane, przesuwania krawędzi, wygładzania punktów,
    znajdowania punktu na dole oraz rasteryzacji.

    Parameters:
        mask_size (tuple):(h, w) Wymiary bitmapy maski.
        example (np.ndarray): Punkty konturu.
        smooth (bool): Jeśli True, wygładza kontur.

    Returns:
        tuple:
            - P1 (np.ndarray): Punkt P1 mitral plane.
            - P2 (np.ndarray): Punkt P2 mitral plane.
            - point_bottom (np.ndarray): Punkt o największej wartości 'y'.
            - points (np.ndarray): Oryginalne punkty konturu.
            - smooth_points (np.ndarray): Wygładzone punkty konturu.
            - mask (np.ndarray): Maska binarna.
    """
    # Wczytanie danych
    points_pure = example

    # Znalezienie mitral plane
    P1, P2, mitral_idx, mitral_plane_distance = find_mitral_plane(points_pure)
    mitral_plane = [P1, P2]

    # Sprawdzenie i przesunięcie krawędzi pomiędzy punktami mitral plane
    points = check_and_shift_edge(points_pure, P1, P2)

    # Wygładzanie punktów, jeśli smooth=True
    if smooth:
        smooth_points = smooth_polygon(points)
    else:
        smooth_points = points

    # Punkt na dole (największa wartość 'y')
    point_bottom = point_of_bottom(smooth_points)

    # Środek odcinka P1-P2
    mid_point = (P1 + P2) / 2

    # Odległość od środka do point_bottom
    vertical_distance = np.linalg.norm(mid_point - point_bottom)

    # Rasteryzacja wielokąta
    mask = rasterize_polygon(smooth_points, mask_size)

    return (
        P1,
        P2,
        point_bottom,
        mitral_plane_distance,
        vertical_distance,
        points,
        smooth_points,
        mask,
    )


def find_contour(mask):
    """
    Wyznacza kontur maski binarnej na podstawie wygładzonego obrazu. Przechodzi przez maskę, wygładza ją,
    a następnie znajduje kontury, przekształcając je do formatu punktów (x, y).

    Parameters:
        mask (np.ndarray): Maska binarna, na której wyznaczany jest kontur.

    Returns:
        tuple:
            - median_smoothed_image (np.ndarray): Obraz po wygładzeniu i denoisingu.
            - point_mask (np.ndarray): Punkty konturu w formacie (x, y).
    """
    # Wygładzanie maski binarnej
    smooth_binary_array = filters.gaussian(mask, sigma=1)

    # Denoisowanie obrazu
    median_smoothed_image = denoise_bilateral(
        smooth_binary_array, sigma_color=0.00005, sigma_spatial=50
    )

    # Znajdowanie konturów
    contours = measure.find_contours(median_smoothed_image, level=0.3)

    # Wybór pierwszego konturu
    chosen_contour = contours[0]

    # Przekształcenie konturu na listę punktów (x, y)
    contour_points = chosen_contour.tolist()
    contour_points_swapped = [(x, y) for y, x in contour_points]

    # Konwersja na tablicę NumPy
    contour_array = np.array(contour_points_swapped)

    # Usuwanie zduplikowanych punktów
    _, unique_indices = np.unique(contour_array, axis=0, return_index=True)
    point_mask = contour_array[np.sort(unique_indices)]

    return median_smoothed_image, point_mask


def min_max_y_point(point_mask):
    """
    Wyznacza punkty o minimalnej i maksymalnej wartości współrzędnej 'y' w zbiorze punktów konturu.

    Parameters:
        point_mask (np.ndarray): Zbiór punktów konturu w formacie (x, y).

    Returns:
        tuple:
            - P1_mask (np.ndarray): Punkty o minimalnej wartości 'y'.
            - point_bottom_mask (np.ndarray): Punkty o maksymalnej wartości 'y'.
    """
    # Znalezienie minimalnej i maksymalnej wartości 'y'
    min_y, max_y = np.min(point_mask[:, 1]), np.max(point_mask[:, 1])

    # Wybranie punktów o minimalnej i maksymalnej wartości 'y'
    P1_mask = point_mask[point_mask[:, 1] == min_y]
    point_bottom_mask = point_mask[point_mask[:, 1] == max_y]

    return P1_mask, point_bottom_mask


def P2_LinearRegression_method(P1_mask, point_mask):
    """
    Przeprowadza regresję liniową na wybranym zbiorze punktów konturu, aby znaleźć punkt P2.
    Punkt P2 jest wyznaczany na podstawie dopasowanej linii oraz punktu P1, który jest punktem wymuszonym.

    Parameters:
        P1_mask (np.ndarray): Punkt wymuszony (P1) na konturze.
        point_mask (np.ndarray): Zbiór punktów konturu w formacie (x, y).

    Returns:
        tuple:
            - reg (LinearRegression): Dopasowany model regresji liniowej.
            - P2_mask (np.ndarray): Wybrany punkt P2 na konturze.
            - P1_mask (np.ndarray): Punkt P1 (wybrany punkt w procesie regresji).
    """
    # Wybór wymuszonego punktu (P1)
    forced_point = P1_mask.ravel()

    # Znalezienie najbliższego punktu w zbiorze
    forced_index = np.argmin(np.abs(np.sum(point_mask - forced_point, axis=1)))

    # Wybór punktów przed i po wymuszonym punkcie
    filter_range = 10
    left_selected = point_mask[max(0, forced_index - filter_range) : forced_index]
    right_selected = point_mask[forced_index + 1 : forced_index + 1 + filter_range]

    # Obliczanie różnicy w 'y' dla punktów po lewej stronie
    y_diff_left = np.abs(np.diff(left_selected[:, 1], append=left_selected[0, 1]))

    # Obliczanie różnicy w 'y' dla punktów po prawej stronie
    y_diff_right = np.abs(np.diff(right_selected[:, 1], append=right_selected[0, 1]))

    # Wybór strony z najmniejszym całkowitym spadkiem
    total_y_diff_left = np.sum(y_diff_left)
    total_y_diff_right = np.sum(y_diff_right)

    if total_y_diff_left <= total_y_diff_right:
        selected_points = left_selected
    else:
        selected_points = right_selected

    # Dopasowanie regresji liniowej
    X = selected_points[:, 0].reshape(-1, 1)
    y = selected_points[:, 1]
    reg = LinearRegression()
    reg.fit(X, y)

    # Przewidywanie wartości y dla punktów
    predicted_y = reg.predict(point_mask[:, 0].reshape(-1, 1))
    differences = np.abs(point_mask[:, 1] - predicted_y)

    # Ustalenie progu dla najmniejszych różnic
    threshold = np.percentile(differences, 21)
    strongly_correlated_points = point_mask[differences <= threshold]

    # Wybranie punktu P2
    P2_mask = strongly_correlated_points[np.argmin(strongly_correlated_points[:, 0])]

    # Wybranie punktu P1
    P1_mask_new = strongly_correlated_points[
        np.argmax(strongly_correlated_points[:, 0])
    ]

    return reg, P2_mask, P1_mask_new


def delete_point_between_P1_P2(P1_mask, P2_mask, point_mask):
    """
    Usuwa punkty pomiędzy dwoma wybranymi punktami P1 i P2 w konturze, zachowując tylko punkty na skrajach.

    Parameters:
        P1_mask (np.ndarray): Punkt P1 w zbiorze punktów.
        P2_mask (np.ndarray): Punkt P2 w zbiorze punktów.

    Returns:
        np.ndarray: Zbiór punktów konturu po usunięciu punktów pomiędzy P1 i P2.
    """
    # Znalezienie indeksów punktów P1 i P2
    idx_p1 = np.where(np.all(point_mask == P1_mask, axis=1))[0][0]
    idx_p2 = np.where(np.all(point_mask == P2_mask, axis=1))[0][0]

    # Określenie zakresu punktów do usunięcia
    start_index, end_index = sorted([idx_p2, idx_p1])

    # Usunięcie punktów pomiędzy P1 i P2
    filtered_points = np.delete(point_mask, np.s_[start_index + 2 : end_index], axis=0)

    return filtered_points


######################################################################################
########################### FINALNA FUNKCJA ##########################################
######################################################################################


def process_mask_to_points(mask):
    """
    Procesuje maskę binarną, analizując jej kontur oraz kluczowe punkty:
    - Wyznaczanie konturu maski.
    - Znalezienie punktów P1 i P2 (definiujących mitral plane).
    - Dopasowanie regresji liniowej do punktów konturu.
    - Usunięcie punktów pomiędzy P1 a P2.
    - Wyznaczenie punktu na dole (point_bottom).
    - Obliczenie odległości mitral plane oraz odległości pionowej od środka do punktu na dole.
    - Wygładzenie konturu i rasteryzacja maski.

    Parameters:
        mask (np.ndarray): Maska binarna reprezentująca obiekt do analizy.

    Returns:
        tuple: Zawiera wyniki analizy maski:
            - mask_2 (np.ndarray): Rasteryzowana i wygładzona wersja maski binarnej.
            - smooth_points_mask (np.ndarray): Wygładzone punkty konturu.
            - P1_mask (tuple): Punkt początkowy mitral plane.
            - P2_mask (tuple): Punkt końcowy mitral plane.
            - point_bottom_mask (tuple): Punkt o największej wartości Y (najniżej położony).
            - mitral_plane_distance (float): Długość mitral plane.
            - vertical_distance (float): Odległość pionowa od środka mitral plane do point_bottom.
            - points_maks_1 (np.ndarray): Filtrowane punkty konturu.
    """
    # Wyznaczenie kontura maski
    median_smoothed_image, point_mask = find_contour(mask)

    # Wyznaczenie punktów P1 i P2
    P1_mask, point_bottom = min_max_y_point(point_mask)

    # Przeprowadzenie regresji liniowej
    reg, P2_mask, P1_mask_new = P2_LinearRegression_method(P1_mask[0], point_mask)

    # Usunięcie punktów pomiędzy P1 a P2
    filtered_points = delete_point_between_P1_P2(P1_mask_new, P2_mask, point_mask)

    # Wyznaczenie z punktów mitral plane, oraz wygładzenie figury
    (
        P1_mask,
        P2_mask,
        point_bottom_mask,
        mitral_plane_distance,
        vertical_distance,
        points_maks_1,
        smooth_points_mask,
        mask_2,
    ) = vector_to_bitmap(filtered_points, mask_size=mask.shape, smooth=True)

    return (
        mask_2,
        smooth_points_mask,
        P1_mask,
        P2_mask,
        point_bottom_mask,
        mitral_plane_distance,
        vertical_distance,
        points_maks_1,
    )
