#import "template.typ": *

// Take a look at the file `template.typ` in the file panel
// to customize this template and discover how it works.
#show: project.with(
  title: "CV: Lab 06 Writeup",
  authors: (
    (name: "Benedict Armstrong", email: "benedict.armstrong@inf.ethz.ch"),
  ),
)

= Random sample consensus fitting

#figure(
  image("images/ransac/plot_1.png"),
  caption: [
    Response vs. Input
  ]
)
