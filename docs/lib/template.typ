#import "@preview/simple-research-poster:0.1.0": *
#import "color.typ": *
#import "tag.typ": *

#show: poster.with(
  title: [Applied Cryonics: How I Slept Through the 21st Century],
  author: [Philip J. Fry and Turanga Leela],
  mentor: [Mentor: Bender B. Rodriguez],
  width: 43in,
  height: 32.5in,
  subtitle: [College of New New York],
  logo: image("test/simple-research-poster/assets/logo.png", height: 150%),
  base-colors: base-colors,
)
#let colored-poster-section = poster-section.with(base-colors: base-colors)

#show: tag.with(
  "NOTE",
  bg_color: indigo-languor.highlight,
  text_color: indigo-languor.cover2,
)

#pad(
  grid(
    columns: 3,
    inset: 0.5in,
    gutter: 30pt,
    [
      #colored-poster-section[Section 1][]
      #colored-poster-section(fill: true)[Section 2][]
    ],
    [
      #colored-poster-section[Section 3][]
      #colored-poster-section(fill: true)[Section 4][]
      #colored-poster-section[Section 5][]
    ],
    [
      #colored-poster-section(fill: true)[Section 6][]
      #colored-poster-section[Section 7][]
      #colored-poster-section(fill: true)[Acknowledgements][]
      #colored-poster-section[References][]
    ],
  ),
  top: 0.5in,
  x: 1in,
)
