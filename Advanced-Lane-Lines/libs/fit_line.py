import numpy as np

class FitLine:
    def __init__(self):

        self.is_first_processed = False

        self.mse_tolerance = 0.01
        self.left_fit = [np.array([False])] 
        self.right_fit = [np.array([False])] 
        
        self.y_eval = 700
        self.ym_per_pix = 3.0/72.0 # meter / pixel(y)
        self.xm_per_pix = 3.7/700.0 # meter / pixel(x)
        self.curvature = 0
        self.ratio = 0.75
       
       
    def update_fit(self, left_fit, right_fit):

        # Handle the first frame
        if self.is_first_processed:
            left_err = ((self.left_fit[0] - left_fit[0]) ** 2).mean(axis=None)
            right_err = ((self.right_fit[0] - right_fit[0]) ** 2).mean(axis=None)
            if left_err < self.mse_tolerance:
                self.left_fit = self.ratio * self.left_fit + (1-self.ratio) * left_fit
            if right_err < self.mse_tolerance:
                self.right_fit = self.ratio * self.right_fit + (1-self.ratio) * right_fit
        else:
            self.right_fit = right_fit
            self.left_fit = left_fit
        
        self.update_curvature(self.right_fit)
     
     
    def update_curvature(self, fit):
        y1 = (2*fit[0]*self.y_eval + fit[1])*self.xm_per_pix/self.ym_per_pix
        y2 = 2*fit[0]*self.xm_per_pix/(self.ym_per_pix**2)
        curvature = ((1 + y1*y1)**(1.5))/np.absolute(y2)
        
        if self.is_first_processed:
            self.curvature = curvature
        
        elif np.absolute(self.curvature - curvature) < 500:
            self.curvature = self.ratio*self.curvature + (1-self.ratio)*(((1 + y1*y1)**(1.5))/np.absolute(y2))

    def get_distance_from_center(self,img):
        if self.right_fit is not None and self.left_fit is not None:
            h = img.shape[0]
            car_position = img.shape[1] / 2
            l_fit_x_int = self.left_fit[0] * h ** 2 + self.left_fit[1] * h + self.left_fit[2]
            r_fit_x_int = self.right_fit[0] * h ** 2 + self.right_fit[1] * h + self.right_fit[2]
            lane_center_position = (r_fit_x_int + l_fit_x_int) / 2
            center_dist = (car_position - lane_center_position) * self.xm_per_pix
        return center_dist

            
        