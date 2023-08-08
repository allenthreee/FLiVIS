import numpy as np
import cv2
import open3d as o3d

# pcd = o3d.io.read_point_cloud("/home/allenthreee/yuanchongjian_ws/data/ftm4cjy/0edited.pcd")
pcd = o3d.io.read_point_cloud("/home/allenthreee/yuanchongjian_ws/data/ftm4cjy/0lidar.pcd")
pcd_arr = np.asarray(pcd.points)  
# print("output array from input list : ", out_arr) 

# Define the camera matrix
fx = 4700.987563588087
fy = 4718.613209287033897
cx = 2115
cy = 1535
# fx = 1364.45
# fy = 1366.46
# cx = 958.327
# cy = 535.074

camera_matrix = np.array([[fx, 0, cx],
						[0, fy, cy],
						[0, 0, 1]], np.float32)

# Define the distortion coefficients
dist_coeffs = np.zeros((5, 1), np.float32)

# Define the 3D point in the world coordinate system
x, y, z = 10, 20, 30
points_3d = np.array([[[x, y, z]]], np.float32)

# Define the rotation and translation vectors
rvec = np.zeros((3, 1), np.float32)
tvec = np.zeros((3, 1), np.float32)
# ycj transformaion
# rvec[0] = 0.5678296*2.1055047
# rvec[1] = -0.580329*2.1055047
# rvec[2] = 0.5837703*2.1055047
# tvec[0] = 0.107
# tvec[1] = -0.0349
# tvec[2] = -0.0067

# manual transformation better result
rvec[0] = 0.5773503*2.0943951
rvec[1] = -0.5773503*2.0943951
rvec[2] = 0.5773503*2.0943951
tvec[0] = 0.0579
tvec[1] = -0.0349
tvec[2] = -0.2067


# Map the 3D point to 2D point
points_2d, _ = cv2.projectPoints(pcd_arr,
								rvec, tvec,
								camera_matrix,
								dist_coeffs)


# image = cv2.imread("/home/allenthreee/yuanchongjian_ws/data/single_scene_calibration/3.png")
image = cv2.imread("/home/allenthreee/yuanchongjian_ws/data/ftm4cjy/0left.png")

count = 0
for point in points_2d:
    # print(point)
	count = count +1
	image = cv2.circle(image, (int(point[0][0]),int(point[0][1])), radius=0, color=(255, 0, 255), thickness=30)

# image = cv2.circle(image, (2000,2000), radius=0, color=(255, 255, 255), thickness=5)

dim = (1280, 720)
# resize image
small_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
cv2.imshow('img',small_image)
cv2.waitKey()
