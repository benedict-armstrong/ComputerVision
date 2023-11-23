#import "template.typ": *

// Take a look at the file `template.typ` in the file panel
// to customize this template and discover how it works.
#show: project.with(
  title: "CV: Lab 05 Writeup",
  authors: (
    (name: "Benedict Armstrong", email: "benedict.armstrong@inf.ethz.ch"),
  ),
)

= Mean-Shift

The mean-shift algorithm can be split into four steps which are repeated until convergence:
- Calculate distances between pixels
- Calculate weights for each pixel based on distance
- Calculate the mean of the pixels weighted by the weights
- Update the pixels to the mean

== Calculating Distances
I simply used numpy's `linalg.norm` function to calculate the distances for all pixels from a given pixel.

== Calculating Weights
We use the Gaussian kernel to calculate the weights for each pixel. The kernel is defined as:
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

Now for the interesting part, the results. I ran the algorithm on the provided image of the ETH main building. I tried a few different bandwidths with 15 steps each to see how they affect the result.

#grid(
  columns: 2,
  column-gutter: 6mm,
  row-gutter: 4mm,
  figure(
  image("images/result_1_14.png"),
  caption: [Mean-shift with bandwidth = $1$]
  ),
  figure(
  image("images/result_3_14.png"),
  caption: [Mean-shift with bandwidth = $3$]
  ),
  figure(
    image("images/result_5_14.png"),
    caption: [Mean-shift with bandwidth = $5$]
  ),
  figure(
    image("images/result_7_14.png"),
    caption: [Mean-shift with bandwidth = $7$]
  )
)

Bandwidths of 1 and 7 seem to be too small and too large respectively. For $b=1$ the image is too noisy and for $b=7$ the image is too smooth and we loose detail such as the left side of the building.

A bandwidth 3 and 5 seems pretty good but somewhere in the middle might be the sweet spot. After playing around abit I found that a bandwidth of 4.5 with 20 steps gives the following (best) result:

#grid(
  columns: 2,
  column-gutter: 6mm,
  figure(
    image("images/result_4.5_19.png"),
    caption: [Mean-shift with bandwidth = $4.5$ and $20$ steps]
  ),
  figure(
    image("images/result_4.5_19_centroids.png"),
    caption: [Mean-shift with bandwidth = $4.5$ and $20$ steps (using color of centroids)]
  )
)

Which seems to be a good balance between the two. There was issue I noticed where `colors.npz` only contained $24$ colors. So if the algorithm generated more than $24$ centroids it would fail. My solution was to simply create a new `colors_2.npz` file with $500$ (random) colors.

=== Color space

#grid(
  columns: 2,
  column-gutter: 10mm,
  figure(
    image("images/scatter_before.png"),
    caption: [3D Scatter plot of the colors before mean-shift]
  ),
  figure(
    image("images/scatter_5_15.png"),
    caption: [3D Scatter plot of the colors after 15 steps of mean-shift with bandwidth = $5$]
  )
)

We can plot the color of each pixel on a 3D scatter plot to see how the colors are grouped. The first image shows the colors before mean-shift is applied, and the second shows the colors after 15 steps of mean-shift with a bandwidth of 5. We can see that the colors are grouped into 5 clusters.

#show link: underline
I've also made a few little #link("https://polybox.ethz.ch/index.php/s/eOBjCNSUK2qIlHO")[GIFs] showing the evolution of the algorithm (including the scatter plot shown above).

