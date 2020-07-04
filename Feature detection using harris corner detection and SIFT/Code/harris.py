try:
    import cv2
    import math
    import numpy as np
    from sklearn.neighbors import KDTree
except Exception as e:
    package = str(e).split()[-1]
    print("ERROR")
    print("{} package not found" .format(package))
    print("Please install the module", package)

def get_interest_points(image, feature_width = 0):

    def apply_filter(image, k_type):
        k_size = 5
        if k_type == "sobel_h":
            image = cv2.Sobel(image, cv2.CV_64F, 1, 0, k_size) 
            
        if k_type == "sobel_v":
            image = cv2.Sobel(image, cv2.CV_64F, 0, 1, k_size) 
            
        if k_type == "gaussian":
            image = cv2.GaussianBlur(image, (k_size, k_size), 1)
        
        return(image)
    
    def calculate_harris_response(Ixx, Iyy, Ixy):
        k = 0.06
        # Calculating determinant
        detA = Ixx * Iyy - Ixy ** 2
        # Calculating trace
        traceA = Ixx + Iyy
        harris_response = detA - k * traceA ** 2
        return(detA, traceA, harris_response)

    
    def find_local_maxima(M, fw = 0, win = 1):
        f_size = fw // 2
        max_resp = np.max(M)
        M = cv2.copyMakeBorder(M, win, win, win, win, cv2.BORDER_CONSTANT, value = -math.inf)
        x_cor = []
        y_cor = []
        response_val = []
        mid_index = ((2*win+1)**2)//2
        for r in range(win + f_size, M.shape[0] - win - f_size):
            for c in range(win + f_size, M.shape[1] - win - f_size):
                mid_val = M[r, c]
                if mid_val >= 0.009 * max_resp:
                    neighbours = M[r - win : r + win + 1, c - win : c + win + 1].flatten()
                    neighbours = np.delete(neighbours, mid_index)
                    if all(list(mid_val > neighbours)):
                        x_cor.append(c - win)
                        y_cor.append(r - win)
                        response_val.append(M[r][c])
        return(np.array(x_cor), np.array(y_cor), np.array(response_val))
           
    def ANMS(x_c, y_c, r_val, limit):
        coordinate = list(zip(x_c, y_c))
        sorted_coordinate = [x for _,x in sorted(zip(r_val, coordinate))]
        sorted_response = sorted(r_val)
        coordinate_radius = []
               
        for i in range(len(sorted_coordinate) - 1):
            x_query = np.array([sorted_coordinate[i]])
            tree = KDTree(np.array(list(sorted_coordinate[j] for j in range(i + 1, len(sorted_coordinate)))))
            radious = tree.query(x_query)[0][0]
            coordinate_radius.append((x_query[0].tolist(), radious[0].tolist()))
        
        sorted_radious = [x for x, _ in sorted(coordinate_radius, key = lambda x : x[1], reverse = True)]

        x_c = list(pair[0] for pair in sorted_radious)
        y_c = list(pair[1] for pair in sorted_radious)
        return(np.array(x_c[:limit]), np.array(y_c[:limit]))
        
        
        
        
        
#     if __name__ == "__main__":
     
    # Calulating spatial derivation
    Ix = apply_filter(image, "sobel_h")
    Iy = apply_filter(image, "sobel_v")

    # Calculating Structure tensor
    Ixx = apply_filter(Ix**2, "gaussian")
    Iyy = apply_filter(Iy**2, "gaussian")
    Ixy = apply_filter(Ix*Iy, "gaussian")

    # Calculating harris response
    detA, traceA, harris_response = calculate_harris_response(Ixx, Iyy, Ixy)
    
    # Finding corner coordinate
    ## Local maxima
    x_coordinate, y_coordinate, response_val = find_local_maxima(harris_response, win = 2, fw = feature_width)
    ## ANMS
    x_coordinate, y_coordinate = ANMS(x_coordinate, y_coordinate, response_val, limit = 1000)
    
    return(x_coordinate, y_coordinate)





