# Vehicles Video Tracking using Computer Vision

This project is about tracking vehicles in a video using computer vision techniques and machine learning. The goal is to **avoid** deep learning techniques.

# Main Problematics to Tackle

- **Scale variation**: cars variate a lot in scale in the dataset
- **Occlusion**: cars are not always visible
- **Deformation**: cars are not always in the same orientation
- **Moving background**: camera is on board of a car

# Techniques to Explore

### Vehicle Detection and Tracking using Computer Vision and Support Vector Machine (SVM)

1. Spacial Bin for Feature Extraction
2. Color Histogram for Feature Extraction
3. Histogram of Oriented Gradients (HOG) for Feature Extraction
4. Apply Support Vector Machine (SVM) on the vehicle and non-vehicle images combined feature vectors to identify the cars in the images in the dataset
5. Draw Rectangle and Heatmap on the Identified Image

# References

### Space-time local features

- **Detectors**:

  - STIP Spatio Temporal Interest Points (Harris3D) [I. Laptev, IJCV 2005] (https://link.springer.com/content/pdf/10.1007/s11263-005-1838-7.pdf)
  - Dollar’s detector [P. Dollar et al., VS-PETS 2005]
  - Hessian3D [G. Willems et al., ECCV 2008]

- **Descriptors**:

  - HOG/ HOF [I. Laptev et al., CVPR 2008] (http://www.irisa.fr/vista/Papers/2008_cvpr_laptev.pdf)
  - Dollar method: Behavior Recognition via Sparse Spatio-Temporal Features [P. Dollar et al., VS-PETS 2005] (https://cseweb.ucsd.edu//~gary/pubs/dollar-vs-pets05.pdf)
  - HoG3D [A. Klaeser et al., BMVC 2008]
  - Extended SURF [G. Willems et al., ECCV 2008] (http://class.inrialpes.fr/pub/willems-eccv08.pdf)

- [2010] MBH: Motion Boundary Histograms (https://hal.inria.fr/inria-00548587/document)
- [2011] Action Recognition by Dense Trajectories (https://hal.inria.fr/hal-00803241/document)
- [2013] Segmentation Driven Object Detection with Fisher Vectors (https://hal.inria.fr/hal-00873134)
- [2014] Accurate Scale Estimation for Robust Visual Tracking(http://www.bmva.org/bmvc/2014/files/paper038.pdf)
- [2015] Online Object Tracking with Proposal Selection (https://hal.inria.fr/hal-01207196/document)
- [2015] EdgeBoxes (https://pdollar.github.io/files/papers/ZitnickDollarECCV14edgeBoxes.pdf)
- Selective Search (https://www.researchgate.net/publication/262270555_Selective_Search_for_Object_Recognition)
- Object Detection with Discriminatively Trained Part Based Models(https://cs.brown.edu/people/pfelzens/papers/lsvm-pami.pdf)

# Data

- http://cogcomp.org/Data/Car/
- GTI: http://www.gti.ssr.upm.es/data/Vehicle_database.html
- KITTI: https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d
- Udacity: https://github.com/udacity/self-driving-car/tree/master/annotations
