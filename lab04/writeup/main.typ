#import "template.typ": *

// Take a look at the file `template.typ` in the file panel
// to customize this template and discover how it works.
#show: project.with(
  title: "CV: Lab 04 Writeup",
  authors: (
    (name: "Benedict Armstrong", email: "benedict.armstrong@inf.ethz.ch"),
  ),
)

// We generated the example code below so you can see how
// your document will look. Go ahead and replace it with
// your own content!

= Assignment
This weeks assignment was split into two parts:
- Implementing a Bag-of-words Classifier
- Implementing a CNN-based Classifier for the CIFAR-10 dataset

= Bag-of-words Classifier
We implemented a bag-of-words classifier using the following steps:
- Generate bag-of-words histogram for each training image
- Find interresting ke
- Given a test image calculate bag-of-words histogram
- Find its nearest neighbor training histogram
- Predict: assign it the category of this nearest training image ($0$/$1$)

== Local Feature Extraction
We start by extracting local features from the images using the SIFT algorithm as discussed in the lecture. 
