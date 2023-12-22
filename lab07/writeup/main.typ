#import "template.typ": *

// Take a look at the file `template.typ` in the file panel
// to customize this template and discover how it works.
#show: project.with(
  title: "CV: Lab 07 Writeup",
  authors: (
    (name: "Benedict Armstrong", email: "benedict.armstrong@inf.ethz.ch"),
  ),
)

= Assignment

For this assignment we were asked to implement a structure from motion (sfm) pipeline and the RANSAC algorithm.


#show link: underline
= Structure from Motion
I started by implementing the `EstimateEssentialMatrix()` function. #link("https://cmsc426.github.io/sfm/", "This") article was very helpful for implementing the 8-point algorithm.

Next I used the `TriangulatePoints` method to determine the correct orientation for the first two cameras.

The final step was matching the keypoints of all the other images to build a 3D point cloud. After some initial trouble I discovered that it worked best when I excluded the image `0000.png`.

The following images were taken from the interactive 3D plot.

#grid(
  columns: 2,
  figure(
  image("images/sfm/sfm3.png"),
  ),
  figure(
  image("images/sfm/sfm4.png"),
  ),
  figure(
  image("images/sfm/sfm5.png")
  ),
  figure(
  image("images/sfm/sfm6.png"),
  ),
)

= Random sample consensus fitting (RANSAC)
Compared to the first task implementing RANSAC was pretty straightforward.

The implementation can roughly be split into four parts:

1. Randomly select a subset of the data. I used `np.random.choice` (without repeating samples) to select the indices of the data points.
2. Find a linear least squares solution to the subset of the data. I used `np.linalg.lstsq` to find the solution.
3. Calculate the outliers and a mask.

#align(
  center,
  ```py
  distances = x * k + b - y
  mask = np.where(distances < thres_dist, True, False)
  ```
)

4. If the number of inliers (`mask.sum()`) is greater than the current best, update the best model.

My results for k, b:
#align(
  center,
  figure(
  table(
  columns: (auto, auto, auto),
  inset: 5pt,
  align: horizon,
  [], [*k*], [*b*],
  [ground truth], [$1$], [$10$],
  [last-squares], [$0.62$], [$8.96$],
  [RANSAC], [$0.99$], [$10.10$],
  ),
  caption: [
    Results for k and b.
  ]
  )
)

// 1 10 
// 0.615965657875546 8.961727141443642 
// 0.9941120730293844 10.101474316875253

#figure(
  image("images/ransac/plot_1.png"),
  caption: [
    Result of RANSAC on the provided data.
  ]
)
