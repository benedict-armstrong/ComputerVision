#import "template.typ": *

// Take a look at the file `template.typ` in the file panel
// to customize this template and discover how it works.
#show: project.with(
  title: "CV: Lab 05 Writeup",
  authors: (
    (name: "Benedict Armstrong", email: "benedict.armstrong@inf.ethz.ch"),
  ),
)

= Assignment
This weeks assignment was split into two parts:
- Implementing mean-shift for image segmentation
- Implementing a simplified version of SegNet

This writeup only covers the first part.

= Mean-Shift

We can split `mean-shift` into five steps:
- Calculate distances between pixels
- Calculate weights for each pixel based on distance
- Calculate the mean of the pixels weighted by the weights
- Update the pixels to the mean
- Repeat until convergence (or number of steps reached)

== Calculating Distances
We simply used numpy's `linalg.norm` function to calculate the distances for all pixels from a given pixel.

== Calculating Weights
We used the Gaussian kernel to calculate the weights for each pixel. The kernel is defined as:
$
  K(x) = e^(frac(x, sqrt(2)b)^2)
$

where $b$ is the bandwidth and $x$ is the distance between the pixels.

== Calculating the Mean and Updating Pixels
The mean can be easily be calculated using the following code:
```py
np.sum(weight.reshape(-1, 1) * X, axis=0) / np.sum(weight)
```

this result is then used to update the pixels.

= Results

Now for the interesting part, the results. We ran the algorithm on the provided image:

#figure(
  image("images/eth.jpg", width: 50%),
  caption: [Original Image]
)


#show link: underline
I've also made fun little #link("https://raw.githubusercontent.com/benedict-armstrong/cv/main/lab05/mean-shift/images/b_5/outline_5.gif")[GIF] showing the evolution of the algorithm.

