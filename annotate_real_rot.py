import cv2
import numpy as np
import os
import math

class KeypointsAnnotator:
    def __init__(self):
        pass

    def load_image(self, img):
        self.img = img
        self.click_to_kpt = {0:"L", 1:"PULL", 2:"PIN", 3:"R"}

    def mouse_callback(self, event, x, y, flags, param):
        cv2.imshow("pixel_selector", self.img)
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.putText(img, self.click_to_kpt[len(self.clicks)], (x,y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)
            self.clicks.append([x, y])
            print (x, y)
            cv2.circle(self.img, (x, y), 3, (255, 0, 0), -1)

    def run(self, img):
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
        return self.get_angle(self.clicks)

    def get_angle(self, points):
        point1, point2 = points
        vec = np.array([point2[0] - point1[0], point2[1] - point1[1]])
        vec_0 = np.array([0, 1])
        angle = np.dot(vec, vec_0)/np.linalg.norm(vec)
        if angle < -math.pi:
            angle += math.pi
        if angle > math.pi:
            angle -= math.pi
        return angle

if __name__ == '__main__':
    pixel_selector = KeypointsAnnotator()

    #image_dir = '/Users/priyasundaresan/Downloads/hairtie_overcrossing_resized'
    #image_dir = '/Users/priyasundaresan/Downloads/overhead_hairtie_random_fabric_resized'
    image_dir = '/Users/jennifergrannen/Documents/Berkeley/projects/rope/six-dof-grasp/datasets/crops_train'

    # image_dir = 'single_knots' # Should have images like 00000.jpg, 00001.jpg, ...
    output_dir = 'real_data_train' # Will have real_data/images and real_data/annots
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    keypoints_output_dir = os.path.join(output_dir, 'annots')
    images_output_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(keypoints_output_dir):
        os.mkdir(keypoints_output_dir)
    if not os.path.exists(images_output_dir):
        os.mkdir(images_output_dir)

    for i,f in enumerate(sorted(os.listdir(image_dir))[::-1]):
        print("Img %d"%i)
        image_path = os.path.join(image_dir, f)
        img = cv2.imread(image_path)
        assert(img.shape[0] == img.shape[1] == 60)
        image_outpath = os.path.join(images_output_dir, '%05d.jpg'%i)
        keypoints_outpath = os.path.join(keypoints_output_dir, '%05d.npy'%i)
        annots = pixel_selector.run(img)
        cv2.imwrite(image_outpath, img)
        print("---")
        annots = np.array(annots)
        np.save(keypoints_outpath, annots)
