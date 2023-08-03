# ICP_Learning-based-6D-Pose-Estimation
Solve the 6D pose estimation with ICP-based(Umeyama with scale calculation) &amp; Learning-based(consider symmetry) methods.

## 1. ICP-based 6D Pose Esimation

Iterative closet points with Umeyama alignment.

### ICP results

<img src="img/1.png" alt="1" style="zoom: 25%;" />

<img src="img/2.png" alt="2" style="zoom: 25%;" />

<img src="img/image-20230803123546590.png" alt="image-20230803123546590" style="zoom:33%;" />

### Improvement Suggestion

Use better Init transformation and Init scale methods.

## 2. Learning-based 6D Pose Estimation

- Direct approaches: predict rotation and translation directly
- Train a neural network: PointNet
- considering symmetry

### Learning-based results

<img src="img/bbox_23_0.png" alt="bbox_23_0" style="zoom:33%;" />

<img src="img/bbox_23_3.png" alt="bbox_23_3" style="zoom:33%;" />

<img src="img/bbox_29_1.png" alt="bbox_29_1" style="zoom:33%;" />
