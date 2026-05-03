## z* = 100 meters, sigma_px = 0.1, outlier=0%, steps=100

Polar3D has the fastest convergence rate, while Invdepth can achieve better accuracy after parallax is large enough.

under pure translation motion, the inclusion of P_vv (velocity covariance for process noise) is necessary to let Euclidean and Polar1D converge. The inclusion of P_UU (vel+rot) seems have no difference compared to P_vv (vel only); we might need to include rotation motion to let it shine.

Before having enough parallax, the inlier ratio was dropping; after that it increases and settles at around 0.9, for all 3 cases of the 1D parametrisations. On the other hand, Polar3D directly takes bearing measurement and starts to converge right after the landmark initialization.


## z* = 100 meters, sigma_px = 0.1, outlier=20%, steps=100

With inlier ratio of 80%, pure Gaussian filter struggles and refuses to converge for all parametrisations, except a slow convergence of Polar3D. The inclusion of P_UU seems to help a bit for the moderate convergence of pure-Gaussian Polar3D.

In the Gaussian-Beta case, all parametrisations converge, although in the end of simulation some would jump to moderately higher error. It seems like Invdepth and Polar3D are more sensitive to noise. The inlier ratio jumps to 0.8 and gradually decreases to 0.5, back and forth relpeatedly, for all parametrisations.

## outlier=40%

At this outlier ratio, even Gauss-Beta filter cannot survive. Strangely, the same inlier ratio fluctuation between 0.5 and 0.8 is still observed here, just with higher fluctuation frequency.

The consistency test with NEES will be our next goal.
