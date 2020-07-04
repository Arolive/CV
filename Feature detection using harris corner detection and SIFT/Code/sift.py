import numpy as np
import cv2
import math
   
def get_features(image, x, y, feature_width, scales=None):
    
    def dic_counter(l):
        return {key : 0 for key in l}
    
    def get_range(val, i):
        return (((val//i)*i, (val//i + 1)*i))
    
    def apply_filter(image, k_type):
        k_size = 5
        if k_type == "sobel_h":
            image = cv2.Sobel(image, cv2.CV_64F, 1, 0, k_size) 
        if k_type == "sobel_v":
            image = cv2.Sobel(image, cv2.CV_64F, 0, 1, k_size) 
        if k_type == "gaussian":
            image = cv2.GaussianBlur(image, (k_size, k_size), 1)
        return image
    
    def calculate_harris_response(Ixx, Iyy, Ixy):
        k = 0.06
        # Calculating determinant
        detA = Ixx * Iyy - Ixy ** 2
        # Calculating trace
        traceA = Ixx + Iyy
        harris_response = detA - k * traceA ** 2
        return(detA, traceA, harris_response)
    
    
    def get_submatrix(matrix, i, j, fw_4):
        return matrix[i : i + fw_4 ,j : j + fw_4]
    
    def get_WTmagnitude_orientation(matrix):
        # Calulating spatial derivation
        Ix = apply_filter(matrix, "sobel_h")
        Iy = apply_filter(matrix, "sobel_v")

        # Calculating Structure tensor
        Ixx = apply_filter(Ix**2, "gaussian")
        Iyy = apply_filter(Iy**2, "gaussian")
        
        mag = apply_filter(np.sqrt(Ixx + Iyy), "gaussian")
        ornt = np.rad2deg(np.arctan2(Iy, Ix))
        ornt = ornt%360
        
        return (mag, ornt)
    
    def fit_parabola(dic):
        count = 0
        sorted_hist = {k: v for k, v in sorted(hist.items(), key=lambda item: item[1], reverse = True)}
        x_mat = []
        y_mat = []
        for i in sorted_hist:
            count += 1
            mid = (i[0] + i[1])/2
            x_mat.append((mid*mid, mid, 1))
            y_mat.append(sorted_hist[i])
            if count == 3:
                break
        return -b/2*a
    
    def create_hist(x, y, binSize):
        interval = (360 - 0)//binSize
        ranges = list((i, i + interval) for i in range(0, 360, interval))
        hist = dic_counter(ranges)
                
        for index in range(len(x)):
            angle = x[index]
            freq = y[index]

            R = get_range(angle, interval)
            midR = (R[0] + R[1])/2
            
            w1 = 1 - abs(midR - angle)/interval
            w2 = 1 - w1
            
            if angle < midR:
                Rn = tuple(number - interval for number in R)
                if Rn[0] < 0:
                    Rn = ranges[-1]
                hist[R] += freq * w1
                hist[Rn] += freq * w2
            elif angle > midR:
                Rn = tuple(number + interval for number in R)
                if Rn[-1] > 360:
                    Rn = ranges[0]
                hist[R] += freq * w1
                hist[Rn] += freq * w1
            else:
                hist[R] += freq
               
        return hist
    
    def get_theta(hist):
        sorted_histVal = sorted(hist.values(), reverse = True)
        max1 = sorted_histVal[0]
        max2 = sorted_histVal[1]

        if max1 >= 0.8 * max2:
            theta = list(hist.keys())[list(hist.values()).index(max1)]
            return (theta[1] - theta[0])/2
        else:
            return fit_parabola(hist)
            
    
    # Main funcion
    
    fw = feature_width
    fw_2 = fw//2
    fw_4 = fw//4
    
    magnitude, orientation = get_WTmagnitude_orientation(image)
    feature = np.empty((0, 128))
    
    for f_no in range(len(x)):
        c, r = x[f_no], y[f_no]
        i1, i2 = r - fw_2, c - fw_2      
        
        sub_ornt, sub_mag = orientation[i1 : i1 + fw, i2 : i2 + fw].copy().flatten(), magnitude[i1 : i1 + fw, i2 : i2 + fw].copy().flatten()

        ornt_mag_hist = create_hist(sub_ornt, sub_mag, binSize = 36)
        theta = get_theta(ornt_mag_hist)
        
        sub_ornt = theta - sub_ornt
        sub_ornt[(fw//2)*(fw + 1)] = theta
        sub_ornt = sub_ornt.reshape(16, 16)
        sub_mag = sub_mag.reshape(16, 16)
        sub_ornt = sub_ornt%360
        
        
        feature_grid = np.array([])        
        for p1 in range(0, fw, fw_4):
            for p2 in range(0, fw, fw_4):
                grid_ornt = get_submatrix(sub_ornt, p1, p2, fw_4).flatten()
                grid_mag = get_submatrix(sub_mag, p1, p2, fw_4).flatten()
                grid_hist = create_hist(grid_ornt, grid_mag, binSize = 8)
                feature_grid = np.append(feature_grid, list(grid_hist.values()))
        feature_grid = feature_grid/np.sum(feature_grid)
        feature = np.append(feature, [feature_grid], axis = 0)
        
    return(feature)
                
                
                
        
        
                
                
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    return fv
###############################################################################################################################
    """
    In this function, you need to implement the SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    -   scales: Python list or tuple if you want to detect and describe features
            at multiple scales

    You may also detect and describe features at particular orientations.

    Returns:
    -   fv: A numpy array of shape (k, feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    # If you choose to implement rotation invariance, enabling it should not    #
    # decrease your matching accuracy.                                          #
    #############################################################################
    '''
    raise NotImplementedError('`get_features` function in ' +
        '`student_sift.py` needs to be implemented')
    '''
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
###############################################################################################################################
    
