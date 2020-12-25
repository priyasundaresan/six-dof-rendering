import cv2
import numpy as np
import os
import math

def gauss_2d_batch(width, height, sigma, u, v, normalize_dist=False):
    x, y = np.meshgrid(np.linspace(0., width, num=width), np.linspace(0., height, num=height))
    x, y = np.transpose(x, (0,1)), np.transpose(y, (0,1))
    gauss = np.exp(-( ((x-u)**2 + (y - v)**2)/ ( 2.0 * sigma**2 ) ) )
    return gauss

def vis_gauss(gaussians):
    output = cv2.normalize(gaussians, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite('test.png', output)

class KeypointsAnnotator:
    def __init__(self):
        pass

    def load_image(self, img):
        self.img = img.copy()
        self.click_to_kpt = {0:"R", 1:"L"}

    def mouse_callback(self, event, x, y, flags, param):
        cv2.imshow("pixel_selector", self.img)
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.putText(self.img, self.click_to_kpt[len(self.clicks)], (x,y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)
            self.clicks.append([x, y])
            print (x, y)
            cv2.circle(self.img, (x, y), 3, (255, 0, 0), -1)

    def run(self, img, img_outpath):
        self.load_image(img)
        self.clicks = []
        cv2.namedWindow('pixel_selector')
        cv2.setMouseCallback('pixel_selector', self.mouse_callback)
        while True:
            k = cv2.waitKey(20) & 0xFF
            if k == 27 or len(self.clicks) == 2:
                break
            if cv2.waitKey(33) == ord('r'):
                self.clicks = []
                self.load_image(img)
                print('Erased annotations for current image')
            if cv2.waitKey(33) == ord('s'):
                self.clicks = []
                print('Skipped current image')
                return None
        self.save_heatmap(img_outpath, img, self.clicks[0])
        return self.clicks[1]

    def save_heatmap(self, img_outpath, img, point):
        # make heatmap output with point
        width, height, _ = img.shape
        cv2.imshow("", img)
        cv2.waitKey(0)
        gauss_sigma = 8
        gauss = gauss_2d_batch(width, height, gauss_sigma, np.array([point[0]]), np.array([point[1]]))
        gauss = gauss.reshape((width, height, 1))
        # vis_gauss(gauss)
        combined = np.append(img, gauss, axis=2)
        np.save(img_outpath, combined)

if __name__ == '__main__':
    pixel_selector = KeypointsAnnotator()

    #image_dir = '/Users/priyasundaresan/Downloads/hairtie_overcrossing_resized'
    #image_dir = '/Users/priyasundaresan/Downloads/overhead_hairtie_random_fabric_resized'
    image_dir = '/Users/jennifergrannen/Documents/Berkeley/projects/rope/six-dof-grasp/datasets/crops_test'

    # image_dir = 'single_knots' # Should have images like 00000.jpg, 00001.jpg, ...
    output_dir = 'real_crop_test' # Will have real_data/images and real_data/annots
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    keypoints_output_dir = os.path.join(output_dir, 'annots')
    images_output_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(keypoints_output_dir):
        os.mkdir(keypoints_output_dir)
    if not os.path.exists(images_output_dir):
        os.mkdir(images_output_dir)

    new_idx = 0
    for i,f in enumerate(sorted(os.listdir(image_dir))[::-1]):
        print("Img %d"%i)
        image_path = os.path.join(image_dir, f)
        img = cv2.imread(image_path)
        assert(img.shape[0] == img.shape[1] == 60)
        image_outpath = os.path.join(images_output_dir, '%05d.npy'%new_idx)
        keypoints_outpath = os.path.join(keypoints_output_dir, '%05d.npy'%new_idx)
        annots = pixel_selector.run(img, image_outpath)
        # cv2.imwrite(image_outpath, img)
        print("---")
        if annots is not None:
            annots = np.array(annots)
            np.save(keypoints_outpath, annots)
            new_idx += 1
        if i > 2:
            break
