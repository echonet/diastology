import numpy as np
import skimage
from scipy.interpolate import splev, splprep
from skimage import filters,measure
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
        ## get disc center: 
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
    a4c_axes_cm = W_PIX*a4c_axes
    if a2c_axes is not None: 
        a2c_axes_cm = W_PIX*a2c_axes
        biplane_axes = list(zip(a4c_axes_cm,a2c_axes_cm))
        volume = np.pi/4.*sum([a[0]*a[1]*h_cm for a in biplane_axes])
    else: 
        volume = sum([np.pi*(a/2.)**2*h_cm for a in a4c_axes_cm])
    return volume

### --- HELPER FUNCTIONS FOR LA SEGMENTATION --- ###
def check_and_shift_edge(points, p1, p2):
    """
    Returns an array of ordered coordinates from points P1
    to P2 of the mitral plane
    """

    return points_new


def find_mitral_plane(points):
    """
    Returns array of coordinates making up the mitral plane
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
    Returns array of coordinates of smoothed polygon
    """
    return smooth_points


def point_of_bottom(points):
    """
    Returns array of coordinates with largest y value 
    """
    # Znalezienie punktu o największej wartości 'y'
    index_of_bottom = np.argmax(points[:, 1])
    bottom_point = points[index_of_bottom]

    return bottom_point


def rasterize_polygon(smooth_points, img_shape):
    """
    Returns array of coordinates in the binary mask
    for a rasterized polygon
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
    Converts a vector of points to a bitmap by: 
    - Determinining the mitral plane with poitns P1 and P2
    - Moving edges 
    - Smoothing points
    - Finding the vertical endpoint of the left atrium 
    - Rasterizing
    """

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
    This function constructs a binary mask from a smoothed image
    """
    return median_smoothed_image, point_mask


def min_max_y_point(point_mask):
    """
    Returns the minimum and maximum y-coordinates for a 
    set of contour points
    """

    return P1_mask, point_bottom_mask


def P2_LinearRegression_method(P1_mask, point_mask):
    """
    Given point P1 of the mitral plane, this function performs
    linear regression to return point P2
    """
    
    return reg, P2_mask, P1_mask_new


def delete_point_between_P1_P2(P1_mask, P2_mask, point_mask):
    """
    This function removes points between points P1 and P2 of the mitral plane
    to isolate the endpoints
    """

    return filtered_points


######################################################################################
########################### FINALNA FUNKCJA ##########################################
######################################################################################


def process_mask_to_points(mask):
    """
    This function generates
    - Point coordinates of the mask
    - Points 1 and 2 of the mitral plane 
    - Horizontal length of the mitral plane 
    - Vertical length of the left atrium 
    """

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
