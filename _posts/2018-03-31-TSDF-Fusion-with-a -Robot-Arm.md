---
layout: post
title: TSDF-Fusion with a  Robot Arm
---



[![alt][3d_recon]](http://people.inf.ethz.ch/moswald/publications/)

<br>

[3d_recon]: assets/img/tsdf/3D_recon.png

The remarkable progress in 2D computer vision and its innumerable applications in recent years has meant that there is a trove of accessible reference material for newcomers to the field. Unfortunately, this is not the case for the some of the more classical (sans deep-deep-neural-nets) approaches in 3D computer vision. These techniques nevertheless remain vitally important to a variety of fields and, in my opnion, will make a come-back in hybrid form with neural-nets in the near future.  

In this blog post we'll talk about one of these fundamental techniques in 3D computer vision: 3D-reconstruction using Truncated Signed Distance Function (TSDF) Fusion. Concretely, we’ll cover the practical details of combining depth images gathered from multiple known camera positions into a 3D surface reconstruction. We’ll also address some of the technical challenges you might come across when dealing with depth images, and present a slightly unconventional way of dealing with them (in 2D). Let’s break down this process into the following components and tackle them one by one:

- [Why TSDF?](#why-tsdf-?)
- [Physical Setup](#physical-setup)
- [Camera Calibration](#camera-calibration)
- [Fusion](#fusion)
- [Experiments with Depth Images](#experiments-with-depth-images)
- [Applications](#applications)

Each section can be read independently without the need to read any of the others as long as you have a general understanding of the concepts therein.

__*N.B.*__ This is not a blog post about implementing TSDF-Fusion. Instead, we’ll focus on practical considerations and setup. You can find excellent open-source [CPU](https://github.com/ethz-asl/voxblox) and [GPU](https://github.com/andyzeng/tsdf-fusion) implementations online.

## Why TSDF?

It is first and foremost important to ask yourself whether TSDF-Fusion is what you need - which implies understanding what it actually is. Technically, in computer-graphics, [SDF](https://en.wikipedia.org/wiki/Signed_distance_function) refers to a representation of a 3D surface on a volumetric ([voxel](https://en.wikipedia.org/wiki/Voxel)) grid, where the value of the function at each voxel approximates its distance from a 3D surface.


TSDF-Fusion can be used as a component in a  [Simultaneous Localisation and Mapping](https://en.wikipedia.org/wiki/Simultaneous_localization_and_mapping) (SLAM) algorithm, which is in-fact the case (amongst other) in the original [KinectFusion paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ismar2011.pdf). In this article however, we ignore the “Localisation” part and focus solely on the fusion of depth images (where each pixel represents the distance to a 3D point) captured from a camera with known extrinsics (6 DoF pose relative to some reference coordinate system).

## Physical Setup

In our case, the set-up consists of a depth camera mounted on a robot-arm. This allows us to determine the camera pose through forward kinematics.

<div>

<table class="table_align" style="width:100%">
    <td class="table_align">
        <img src="assets/img/tsdf/physical_setup_small.png" alt="Physical Setup">
    </td>
</table>
</div>

*Figure 1: Physical Setup. The pointer is used to localise the position of the calibration rig as discussed in the next section.*

[physical_setup]: assets/img/tsdf/physical_setup_small.png

We use a PMD [pico flexx](https://pmdtec.com/picofamily/flexx/) depth camera fixed onto the robot-arm using a 3D-printed mount. The pico flexx uses time-of-flight technology, which although noisy from one frame to the next, yields relatively accurate point-clouds. The table below compares the pico flexx against other depth cameras commonly used in the research community.


| Camera Name  | Technology | Depth Range (m)| Frame Rate (fps)| Size (mm)|
|---|:---:|:---:|:---:|:---:|
| Pico Flexx  | Time-of-Flight  |  0.1 - 7 |  5 - 45 | 68 x 17 x 7.35 |
| Intel Realsense D435| Active IR  | 0.2 - 10 | 30 - 90 | 90 x 25 x 25 |
| Primesense  Carmine| -  |  0.3 - 3.5 | 30  | 180 x 25 x 35|


Our use case in particular requires operating between 0.15 - 3 meters from the object surface which, along with its small form factor, is the primary motivator for pico flexx. It however does not come with an in-built RGB camera, so you'd need to do a manual calibration (which might be a deal breaker for some).

## Camera Calibration

Whether or not you’re using the pico flexx, it is essential to have a good intrinsics and extrinsics calibration of your camera. This amounts to estimating the $$\mathbf{K}$$ and $$\mathbf{T}$$ matrix in the camera projection equation below, which transforms a 3D point $$\mathbf{w}$$ (specified in homogenous coordinates) from some arbitrary reference frame to 2D pixel coordinates $$\mathbf{u}$$ in the camera frame.

$$

\lambda

\begin{bmatrix}

u \\
v \\
1

\end{bmatrix}

=

\underbrace{

     \begin{bmatrix}
     \phi_x & \gamma & \delta_x  & 0  \\
     0 & \phi_y & \delta_y  & 0  \\
     0 & 0 & 1  & 0  \\

     \end{bmatrix}
}_\mathbf{K}

\underbrace{

\begin{bmatrix}
\omega_{11} & \omega_{12} & \omega_{13}  & \tau_x  \\
\omega_{21} & \omega_{22} & \omega_{23}  & \tau_y  \\
\omega_{31} & \omega_{32} & \omega_{33}  & \tau_z  \\
0 & 0 & 0  & 1  \\

\end{bmatrix}
}_\mathbf{T}

\begin{bmatrix}

x \\
y \\
z \\
1

\end{bmatrix}
\tag{1}\label{eq:one}


$$


### Intrinsics

Intrinsics calibration refers to the estimation of the [camera matrix](http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/parameters.html) $$\mathbf{K}$$ which accounts for the projection model of the pinhole-camera. It takes into account the the distance and alignment of the image plane relative to the camera optical center ($$F_c$$ in Figure 2). A 3D point $$\mathbf{w}=(x, y, z)$$ in camera coordinate frame $$F_c$$,  projects onto the camera image plane at pixels $$\mathbf{u}=(u, v)$$.


$$

u = \frac{\phi_x x + \gamma y}{z} + \delta_x

\tag{2} \label{eq:two}
$$ 

<br>

$$
v = \frac{\phi_y y}{z} + \delta_y

\tag{3}\label{eq:three}
$$ 

<br>

Applying the camera matrix in equation 1 above normalises the camera model: yields focal length ($$\phi_x$$ and $$\phi_y$$) 1, offsets ($$\delta_x$$ and $$\delta_y$$) the location of the optical centre relative to the top left corner of the image, and corrects for manufacturing “defects” such as irregular photoreceptor spacing and skew ($$\gamma$$).


<div>

<table class="table_align" style="width:100%">
    <td class="table_align">
        <img src="assets/img/tsdf/pin_hole_model.png" alt="Pinhole camera model">
    </td>
</table>
</div>

*Figure 2: Pinhole Camera Model.*


In addition to the projection parameters, the formation of an image on the camera’s pixel array is also affected by the “focusing” effect of the lens - which a pinhole model ignores. These can be summarised with a set of radial $$(k_1, k_2, k_3)$$ and tangential $$(p_1, p_2)$$ [distortion coefficients](https://www.mathworks.com/help/vision/ug/camera-calibration.html#bu0nj3f). Is applied after normalising (perspective projection $$\mathbf{x \rightarrow}\mathbf{x'}$$) the 3D point but before the pin-hole projection (the action of $$\mathbf{K}$$).

$$

\begin{bmatrix}

x' \\
y' \\

\end{bmatrix}
=
\begin{bmatrix}

x/z \\
y/z \\

\end{bmatrix}

$$

<br>

$$

\hat x = \underbrace{
          (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) x'}_{Radial Component} +
          \underbrace{
          2 p_1 x' y' + p_2 (r^2+ 2 x'^2)}_{Tangential Component}

\tag{4}\label{eq:four}

$$ 

$$

\hat y = \overbrace{
          (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) y'} +
          \overbrace{
          p_1 (r^2+ 2 x'^2) + 2 p_2 x' y'}

\tag{5}\label{eq:five}

$$ 

<br>

You can now get the pixel projections $$(u, v)$$ by replacing $$(x, y)$$ in equation $$\eqref{eq:two}$$ and $$\eqref{eq:three}$$ by $$(\hat x, \hat y)$$. The problem of intrinsics calibration therefore is to estimate the following parameters: {$$\phi_x, \phi_y, \delta_x, \delta_y, \gamma, k_1, k_2, k_3, p_1, p_2$$}.


Intrinsics calibration is a fairly standard procedure; you can find a step-by-step tutorial [here](https://docs.opencv.org/3.4.3/dc/dbb/tutorial_py_calibration.html).
Most calibration algorithms presume images taken from multiple, varied vantage points where known world coordinates can be identified reliably (such as a the corners of chessboard pattern). On the pico flexx, we can do this by accessing the intensity image *Figure 3b*. The pixel values here represent the magnitude of “excitement” of the photoreceptors, which you’ll need to normalise into a range that libraries such as OpenCV expect (```CV_8U```, ```CV_16U```, etc.)


<div>

<table class="table_align">
    <tr>
        <td class="table_align">
            <img src="assets/img/tsdf/depth_image.png" alt="Depth Image"> <br>
             <i>(a)</i>
        </td>
        <td class="table_align">
            <img src="assets/img/tsdf/calib_pattern_intensity.png" alt="Intensity Image"> <br>
             <i>(b)</i>
        </td>
    </tr>
</table>
</div>  

*Figure 3: (a) A depth image taken from the pico flexx, (b) An intensity image of a calibration target (chessboard) with the detected corners highlighted and sorted.*

Once you have a set of calibration `images`, the procedure can be summarised in the following pseudo-code snippet.

```python
# Intrinsics Calibration
points_3d = zeros([board_size ** 2, 3])
points_3d[:, :2] = mgrid(:board_size, :board_size).T.reshape(-1, 2) * square_size

for image in images:
    points_2d = cv2.findChessboardCorners(image, (board_size, board_size))

    if not points_2d:
        continue

    img_points.append()
    world_points.append(points_3d)

cam_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(world_points, img_points)

# Optional Extra Extrinsics Calibration
T_cb_cams = []

for world_point, img_point in zip(world_points, img_points):
    rvec_cb_cam, tvec_cb_cam = cv2.solvePnPRansac(world_point, img_point)

    # Convert to 4x4 transformation matrix
    T = combine(cv2.Rodrigues, tvec_cb_cam)
    T_cb_cams.append(T)
```


### Extrinsics

Unlike the case for a SLAM problem, we do not want to estimate the global pose of the camera at every time step. Instead, we want to estimate the extrinsic transformation of the camera’s optical center relative to its mounting point on the robot arm.

At first glance, having a 3D CAD model of the camera-mount might appear to obviate the need for such a calibration. Unfortunately, in practice even small misalignments compound as a function of distance when rotations are involved as visualised by *Figure 4* below.


![alt The concept behind TSDF fusion ][rotation_compounding]

[rotation_compounding]: assets/img/tsdf/rotation_compounding.png

*Figure 4: The compounding effect of small misalignments on rays of light.*


The ```cv2.calibrateCamera``` method from the code snippet above simultaneously solves for both intrinsics and extrinsics, which are returned in axis-angle representation (`rvecs`, `tvecs`). In practice you get slightly better results if you carry out a second calibration of extrinsics. In particular, you can now use [RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus) and ignore the possible effect of outliers.

### Extrinsics Kinematic Chain

Solving the extrinsics optimisation gives us the 6 DoF transformation between the camera and the origin of the chessboard pattern $$T^{cb}_{cam}$$. We still do not know how the camera (which is so far floating around in space) is orientated with respect to the robot arm $$T^{cam}_{ra}$$. The missing link is pose of the chessboard with respect to the global (robot) reference frame $$T^{cb}_W$$. Once we know this transformation, estimating the camera pose with respect to this reference frame simply amounts to a chain of rigid transformations:

$$
T^{cam}_{W} = \underbrace{
               T^{ra}_W \cdot T^{cb}_{ra}
               }_{T^{cb}_{W }}
               \cdot T^{cam}_{cb}
$$

<br>

Where the notation $$T^a_b \cdot \vec{v}$$ denotes the rigid 6-DoF transformation required to transform a vector $$\vec{v}$$ from reference frame $$a$$ to an arbitrary frame $$b$$.

You could either have a (very) precise measurement of $$T^{cb}_W$$ by placing the pattern at a pre-calibrated position, or measure $$T^{cb}_{ra}$$ using the robot as a pointing device. In this case, we mount a 3D printed pointer on the robot and point it at the origin of the chessboard pattern, while making sure to align the en-effector to match the coordinate axes of the chessboard.

<div>
<table class="table_align" style="width:100%">
<td class="table_align">
    <img src="assets/img/tsdf/calib_setup.png" alt="Calibration Setup">
</td>
</table>
</div>  

*Figure 5: Calibration setup. The coordinates of the chessboard $$T^{cb}_W$$ can be determined using a pointer with known forward kinematics $$T_{ra}$$*

*Practical Considerations:*
- Make sure to image the chessboard from a range of view points.
- If you have a lot of lens distortion, you can either use the fisheye calibration method in OpenCV, or you could only consider a central crop of the depth image. This is the case for the pico flexx, the central crop works well but the edges show large distortions.
- In case you crop your images, remember to compensate the offset parameters $$(\delta_x, \delta_y)$$ in the Camera Matrix.

We now finally have all the ingredients to start fusing some TSDFs!

## Fusion

While we’re not going to go into the implementation details of TSDF-Fusion, let’s have a quick theoretical overview to understand what’s happening under-the-hood, so that you can debug when things go awry.

As we briefly discussed in the first section, our aim is to combine a number of depth images taken from known camera poses into a 3D reconstruction.  Since images taken from different vantage points might not align exactly (*Figure 11b*), we take the weighted average of multiple independent surface measurements. The [original paper](https://graphics.stanford.edu/papers/volrange/volrange.pdf) introducing the technique gives a relatively comprehensible explanation.

![alt The concept behind TSDF fusion ][tsdf_concept]
*Figure 6: Visualing the concept behind averaging two measured surfaces to fuse them into one. Curless, B., & Levoy, M. (1996)*

[tsdf_concept]: assets/img/tsdf/tsdf_concept.png




The $$T$$ in TSDF becomes relevant when computing such a surface representation from a collection of “noisy” depth images. It has to do with the “truncation distance” $$\epsilon$$ behind a surface, within which we assume a depth measurement to originate from it. It is introduced to minimise the possibility of surfaces interacting when imaged from opposite directions, which becomes particularly important for relatively thin objects. *Figure 7* visualises this concept, where the truncation margin around the inner side of the surface is highlighted in red.

<!-- | ![alt The concept behind TSDF fusion ][tsdf_truncation_concept] |
|:--:| -->

![alt Truncation concept ][tsdf_truncation_concept]
*Figure 7: Each voxel $$v$$ in a discrete voxel-grid contains its distance $$d^v$$ from the nearest surface. Depth measurement from the camera a truncated to a distance $$\epsilon$$ from the surface.*

[tsdf_truncation_concept]: assets/img/tsdf/tsdf_truncation_concept.png

Each voxel $$v$$ on the grid contains its distance $$d^v$$ from the nearest surface. One way to compute the $$d^v$$ is to iterate through the entire voxel-grid, check whether each voxel is visible in a given camera frame and calculate:

$$
d^v = I_d - v_{cam}
$$

<br>

where $$v_{cam}$$ is the distance of the voxel from the camera.

With that in mind, the essence of the TSDF algorithm is summarised in the following four equations:

$$

\renewcommand{\vec}[1]{\mathbf{#1}}

D(\vec{x}) = \frac{\sum w_{i}(\vec{x}) d_{i}(\vec{x})}
			{\sum w_{i}(\vec{x})}

\tag{6}\label{eq:six}

$$ 

<br>

$$
\renewcommand{\vec}[1]{\mathbf{#1}}

W(\vec{x}) =  w_i(\vec{x})

\tag{7}\label{eq:seven}

$$ 

<br>

$$

\renewcommand{\vec}[1]{\mathbf{#1}}

D_{i+1}(\vec{x}) = \frac{W_i(\vec{x}) D_i(\vec{x}) + w_{i+1}(\vec{x})d_{i+1}(\vec{x})}{W_i(\vec{x}) + w_{i+1}(\vec{x})}

\tag{8}\label{eq:eight}

$$

<br>

$$
\renewcommand{\vec}[1]{\mathbf{#1}}

W_{i+1}(\vec{x}) = W_{i}(\vec{x}) + w_{i+1}(\vec{x})

\tag{9}\label{eq:nine}

$$

<br>

Equation $$\eqref{eq:six}$$ gives the combination rule for the cumulative signed-distance function $$D(\mathbf{x})$$ that we’re trying to estimate where:

- $$d_i(\mathbf{x})$$ is the signed-distance map at each time-step (computed from the depth image).
- $$w_i(\mathbf{x})$$ is the weight function for the current time-step.
- $$W(\mathbf{x})$$ is the cumulative weight function - it takes into account all the measurements so far.

The weight function serves two roles:

1. It weighs successive depth measurements against the cumulative sum of all previous measurements. This serves to consolidate measurements that agree, while allowing incremental updating with new data.
2. It allows us to model camera specific measurement uncertainty. That is, how a particular sensing technology (such as Time-of-Flight) affects depth measurement. The reliability of depth estimation for instance, might drop as a function of the distance from the camera’s optical centre (due the effects of lens distortion etc). A way to overcome this could be to design the weighting function to be a 2D Gaussian centered at the optical center.

Although certain sensing modalities might be more susceptible to unmodelled camera-specific measurement uncertainty, in practice, you can get reasonable results even if you ignore the second point - *i.e.* assume $$w_i(\mathbf{x})$$ to be uniformly distributed (equal to $$1$$ at each step).


Equations $$\eqref{eq:eight}$$ and $$\eqref{eq:nine}$$ give the update rules for the cumulative signed-distance and weight functions for each new frame (time step $$i+1$$).

That's it! Combining all the steps mentioned so far allows us to compute a 3D reconstructed point-cloud. The pseudo-code below summarises the algorithm.

{% highlight python %} 
for idx, voxel, d_v in enumerate(grid, distances):

    if not voxel_in_frame(idx):
        continue

    old_weight = weight_array[idx]
    new_weight = old_weight + 1     # uniform weight distribution
    d = min(1, d_v / epsilon)       # truncated distance to surface

    # Update
    tsdf_array[idx] = (tsdf_array[idx] * old_weight + d) / new_weight
    weight_array[idx] = new_weight
{% endhighlight %}

The resulting point-cloud can be turned into a mesh like the one in *Figure 13* using the [marching-cubes algorithm](http://paulbourke.net/geometry/polygonise/).

![][3D_tsdf]

*Figure 8: 3D mesh reconstructed with TSDF-Fusion.*

[3D_tsdf]: assets/img/tsdf/3D_TSDF.png

*Practical Considerations:*
- Implementing a discrete voxel-grid based method is likely to be quite slow and require a GPU. For faster approaches you can look into [sparse voxel octrees](https://www.nvidia.com/object/nvidia_research_pub_018.html).
- The quality of the reconstruction greatly depends on your extrinsics calibration, and the viewing angle relative to a surface. The noise covariance ellipses of most depth cameras expand around regions tangential to viewing direction.



## Experiments with Depth Images

Sometimes it might be necessary to perform warping operations (resize, rotate, distort) on depth images where, unlike RGB images, the pixel values have a spatial meaning. An interpolation operation between adjacent pixels therefore cannot simply be expressed as a bi-linear (or bi-cubic, *etc.*) average.

This becomes problematic when interpolating around regions with dead pixels (pixels which have no corresponding depth measurement) or object edges. Dead pixels are common in almost every type of depth sensing modality; they appear around edges due to occlusions in stereo-imaging, on objects that are black in IR imaging (since black is a good sink for IR radiation), *etc.*  

Since depth images are actually just point-clouds, the common way of dealing with them is by performing operations in 3D. Libraries such as [PCL](http://www.pointclouds.org/) provide a number of convienient methods, such [bilateral upsampling](http://docs.pointclouds.org/1.7.0/classpcl_1_1_bilateral_upsampling.html#details), to do such operations.

Here we'll consider an alternative approach by working with the "projection" of the point-cloud onto a 2D image. In particular, this approach is useful when generalising image pre-processing to depth images in a deep-neural-network pipeline. Some advantages of doing this are:

- You can operate directly on the 2D depth image, rather than converting it to a point-cloud format.
- It allows us to perform arbitrary warping operations on depth images using pipelines designed for mono-images.
- It allows us to construct a minimal solution that only addresses the interpolation part and leaves the mapping/filtering/smothing operation up to the user.
- Given a 2D image, you can design an algorithm with $$O(N)$$ complexity, where $$N$$ is the output image size.
- It is trivial to parallelise.


__*N.B.*__ If you're doing conventional 3D computer vision, you're probably better off in the long-run to manage point-cloud operations in 3D with PCL *etc*. The method mentioned below is particularly designed for a deep-learning setting where you might want image warping as an augmentation technique, although you still need to keep track of it (since it affects the 3D position of points).


Since the image pixels still have spatial meaning though, using out-of-the-box interpolation techniques developed for color images doesn't work and leads to undesired "flying pixels" in the processed image and 3D-reconstruction *Figure 9*.


<div>
<table class="table_align" style="width:100%">
    <tr>
        <td class="table_align">
            <img src="assets/img/tsdf/int_linear.png" alt="Bilinear Interpolation"> <br>
             <i>(a)</i>
        </td>
        <td class="table_align">
            <img src="assets/img/tsdf/linear_recon.png" alt="Linear Reconstruction"> <br>
             <i>(b)</i>
        </td>
    </tr>
</table>
</div> 

*Figure 9: (a) Naive bi-linear interpolation on a depth image results is flying pixels that look like fuzzy edges in 2D. (b) TSDF reconstruction makes the flying pixels more obviously visualisable.*

Probably the easiest option is to use a nearest-neighbour interpolation, which should suffice for most cases, but performs somewhat poorly around edges, which become more jagged and imprecise, since you're rounding off pixel coordinates. Additionally, noise present in the original also gets carried over to the warped image.

Instead, let's experiment with a custom algorithm for interpolating depth images, that:

- Ignores regions with dead-pixels.
- Elegantly handles interpolation around regions with object edges.
- Smooths areas with outlier (noisy) pixel while maintaining edge boundaries (essentially a bilateral filter).

We’ll consider the case of bi-linear interpolation; a quadratic estimation of the value of fractional pixel coordinates $$P$$ based on the values of its four bounded-box pixels {$$F_{00}, F_{01}, F_{10}, F_{11}$$}.


![alt Bilinear Interpolation][bilinear_int]

*Figure 10: Bilinear Interpolation.*

[bilinear_int]: assets/img/tsdf/bilinear_int.png


First, we need a mapping of pixels from the source (original) to the destination (warped) image. As an example, we can use the OpenCV ```initUndistortRectifyMap``` function, which would normally be used to correct for lens distortion on monocular images. It returnes a handy map (```cmap``` in the code below) which tells us where each pixel in the destination image gets its value from in the source image, an operation that rarely yields integer pixel coordinates.

__*N.B.*__ You normally do image undistortion *before* computing depth maps. Here we're using it purely as an convenient example.

#### Ignoring Dead-pixels

Given ```cmap``` we first ignore all pixel that lie outside the image boundaries. Next, we look up the value of each of the four bounding-box pixels. If two or more of these four pixels are zero, we consider this point to lie near a dead-pixel zone and assign a value of zero.

#### Edge Case (literally)

Next we consider regions that lie around object edges. To detect these, we compute the [*Laplacian*](https://en.wikipedia.org/wiki/Discrete_Laplace_operator) of the source image (```cv::Laplacian```). Technically the Laplacian is the divergence of the gradient at each pixel but it is approximated to a discrete 2D grid using a convolution with a 3 x 3 kernel:

$$
\nabla^2 =
     \begin{bmatrix}
     0 & 1 & 0  \\
     1 & -4 & 1  \\
     0 & 1 & 0  \\
     \end{bmatrix}
$$

<br>

It acts by highlighting areas with rapid change in image gradients. This happens twice around each edge pixel; moving from a region with small gradients (non-edge pixels) to an area with high gradients (edge pixels) and back again (non-edge pixels). Depending on the relative pixel intensities on either side of an edge, the Laplacian yields two edges of pixels with opposite signs (*Figure )

<div>
<table class="table_align" style="width:100%">
    <tr>
        <td class="table_align">
            <img src="assets/img/tsdf/laplacian.png" alt="Edge Image"> <br>
             <i>(a)</i>
        </td>
        <td class="table_align">
            <img src="assets/img/tsdf/edge_image.png" alt="Laplacian Image"> <br>
             <i>(b)</i>
        </td>
    </tr>
</table>
</div> 

*Figure 11:(a) Zoomed in image of an edge between two objects. (b) When moving across an edge, the Laplacian highlights two edges, one on each image. The signs and magnitudes of the pixel values at these edges depend on the relative pixel intensities when moving across the edge.*



This doubly highlighted edge as result of the Laplacian allows us detect pixels belonging to objects on either side of the edge boundary and assign them to the correct one.

To visualise this point better, consider the case in *Figure 12*. It visualises the four possible cases that might arise when interpolating around edge pixels.


![alt Edge Interpolation Concept ][edge_aware_concept]

*Figure 12: Visualising interpolation around edge pixel belonging to two objects: A (Red) and B (Green). In each case, the interpolated point in the source image $$p$$ is marked with and x.*

[edge_aware_concept]: assets/img/tsdf/edge_aware_concept.png


In *Figure 12a*, three of the four bounding-box pixels lie on an edge. The top-left pixel $$p_1$$ belongs to the edge of object $$A$$ while the other two pixels belong to that of object $$B$$. To detect this and assign the pixel to $$B$$, we threshold the difference between the median values of the pixels and assign $$p$$ to the average of $$p2$$ and $$p3$$. We can do this because the pixel values at {$$p_1, p_2, p_3, p_4$$} have spatial meaning and therefore their difference indicates the distance between them. We can therefore decide for instance, that edge pixeles that are withing 1 $$cm$$ belong to the same object.

A second edge case is where a horizontal or vertical edge boundary divides the bounding-box equally (*Figure 12c-d*). In this case, we finally give up on interpolation (because there isn’t really a correct one!) and assign $$p$$ to its nearest neighbour.

The code for implementing all of this is given below.

{% highlight C++ %} 
using namespace std;
using namespace cv;

void edgeAwareRemap(InputArray _src, OutputArray _dst, InputArray _cmap)
{
    int x0, y0, x1, y1, edgeThreshold;
    float x, y, f00, f01, f10, f11, fxy;
    vector<bool> isEdgePx(4);
    vector<float> pxList(4);

    Mat_<float> src = _src.getMat(), dst = _dst.getMat();
    Mat cmap = _cmap.getMat();

    int h = dst.rows, w = dst.cols;

    Mat_<float> laplacianImage;
    Mat_<float> yMat(1, 2, CV_32F), xMat(2, 1, CV_32F), F(2, 2, CV_32F);

    Laplacian(src, laplacianImage, CV_32F, 1);
    edgeThreshold = 400;

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {

            isEdgePx.clear();

            y = cmap.at<Point2f>(i, j).y;
            x = cmap.at<Point2f>(i, j).x;

            // Ignore invalid indices
            if ((((h - 1) < y) || (y < 0)) ||
                (((w - 1) < x) || (x < 0))) {
                dst(i, j) = 0.0;
                continue;
            }
            else {
                y0 = static_cast<int>(floor(y));
                x0 = static_cast<int>(floor(x));
                y1 = static_cast<int>(ceil(y));
                x1 = static_cast<int>(ceil(x));

                f00 = src.at<float>(y0, x0);
                f01 = src.at<float>(y0, x1);
                f10 = src.at<float>(y1, x0);
                f11 = src.at<float>(y1, x1);

                isEdgePx.push_back((edgeThreshold < laplacianImage(y0, x0)) ||
                    (laplacianImage(y0, x0) < -edgeThreshold));
                isEdgePx.push_back((edgeThreshold < laplacianImage(y0, x1)) ||
                    (laplacianImage(y0, x1) < -edgeThreshold));
                isEdgePx.push_back((edgeThreshold < laplacianImage(y1, x0)) ||
                    (laplacianImage(y1, x0) < -edgeThreshold));
                isEdgePx.push_back((edgeThreshold < laplacianImage(y1, x1)) ||
                    (laplacianImage(y1, x1) < -edgeThreshold));

                if (any_of(isEdgePx.begin(), isEdgePx.end(), [](bool v)
                { return v; })) {
                    pxList = {f00, f01, f10, f11};
                    sort(pxList.begin(), pxList.end());

                    if ((pxList[2] - pxList[1]) < 20) {
                        // Assign median value
                        dst(i, j) = static_cast<int>((pxList[1] + pxList[2]) / 2);
                        continue;
                    }
                    else {
                        // Find and assign to nearest neighbour
                        dst(i, j) = src(static_cast<int>(round(y)),
                                        static_cast<int>(round(x)));
                        continue;
                    }
                }

                // Tolerate at-most one zero value for corner pixels
                if ((f00 == 0) && (f01 != 0) && (f10 != 0) && (f11 != 0)) {
                    f00 = (f00 + f11) / 2;
                }
                else if ((f00 != 0) && (f01 == 0) && (f10 != 0) && (f11 != 0)) {
                    f01 = (f00 + f11) / 2;
                }
                else if ((f00 != 0) && (f01 != 0) && (f10 == 0) && (f11 != 0)) {
                    f10 = (f00 + f11) / 2;
                }
                else if ((f00 != 0) && (f01 != 0) && (f10 != 0) && (f11 == 0)) {
                    f11 = (f01 + f10) / 2;
                }
                else if (!((f00 != 0) && (f01 != 0) && (f10 != 0) && (f11 != 0))) {
                    dst(i, j) = 0;
                    continue;
                }

                // Interpolate
                yMat << y1 - y, y - y0;
                xMat << x1 - x, x - x0;
                F << f00, f01, f10, f11;
                fxy = Mat_<float>(yMat * F * xMat)(0, 0);
                dst(i, j) = uint16_t(fxy);
            }
        }
    }
};
{% endhighlight %}



The results of applying such a remapping operation are visualised in (*Figure 13b*). In comparison, the nearest neighbour interpolation (*Figure13c*) smoothed with a median filter has pronounced distortions at the edges (*Figure13d*) when compared to the edge-aware interpolation. The median filter dialates already distorted edges, creating spurious data.

Our custom interpolation on the other-hand strikes a balance between edge-preserving warping, smoothing and accurate interpolation.


<div>
<table class="table_align" style="width:100%">
    <tr>
        <td class="table_align">
            <img src="assets/img/tsdf/raw_depth.png" alt="Raw Depth Image"> <br>
             <i>(a)</i>
        </td>
        <td class="table_align">
            <img src="assets/img/tsdf/raw_eaw.png" alt="Edge Aware Interpolation"> <br>
             <i>(b)</i>
        </td>
    </tr>
    <tr>
        <td class="table_align">
            <img src="assets/img/tsdf/Interpolated_NN.png" alt="Nearest Neighbour"> <br>
             <i>(c)</i>
        </td>
        <td class="table_align">
            <img src="assets/img/tsdf/diff_image_median.png" alt="Diff Image"> <br>
             <i>(d)</i>
        </td>
    </tr>
</table>
</div> 

*Figure 13: Comparison of nearest-neighbour (c) and the custom interpolation (b) techniques from a warped raw image (a). (d) Difference image highlighting edge discrepancies between (b) and (c)*

This is ofcourse not the only way you could do this. Here we've approximated the interpolation with a median/nearest-neighbour approach when the source pixel lies on a edge. You could probably do better by using some other kind of average. But unless you really care about those diminishing returns, after a point the difference will be imperceptible.

*Practical Considerations:*

- real returns will depend on the type of image (big features or relatively smooth)
- might want to benchmark more with smaller objects (where it is likely to be more useful)

## Applications

While there have been leaps of progress in 2D computer vision, 3D vision has lagged behind mostly due to a lack of tools to deal with its geometric nature. Yet, the world is 3D and there are obvious advantages of treating it as such.

[Geometric Deep Learning](http://geometricdeeplearning.com/) has a list of the state of the art methods in deep-learning on graphs, a few of which include some pretty impressive work on 3D vision. here are a few examples on the exciting use of 3D vision:

- Shape segmentation with [SyncSpecCNN](https://arxiv.org/pdf/1612.00606.pdf):

![syncspeccnn][syncspeccnn]

[syncspeccnn]: assets/img/tsdf/syncspeccnn.png

- Dense shape correspondence with [Deep Functional Maps](https://arxiv.org/pdf/1704.08686.pdf)

![deep_functional_maps][deep_functional_maps]

[deep_functional_maps]: assets/img/tsdf/deep_functional_maps.png


- Learning visual descriptors for object using [Dense Object Nets](https://arxiv.org/pdf/1806.08756.pdf):

![dense_visual_descriptors][dense_visual_descriptors]

[dense_visual_descriptors]: assets/img/tsdf/dense_visual_descriptors.png




- Directly learning Signed Distance Functions with [DeepSDF](https://arxiv.org/pdf/1901.05103.pdf):

![deepsdf][deepsdf]

[deepsdf]: assets/img/tsdf/deepsdf.png


3D computer vision is an exciting and important field. It deserves at least as much attention as 2D vision if not more! I hope this blog post helps a few more people approach it :)
