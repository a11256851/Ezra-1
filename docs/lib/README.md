# Documentation Library & Style Guide

This directory (`/project/docs/lib/`) serves as the core design system for our project's documentation. We use **Typst** for typesetting to ensure professional, consistent, and high-performance document generation.

## Directory Structure

- `color.typ`: Centralized color palettes and theme definitions.
- `README.md`: Usage guidelines and documentation standards (this file).
- `template.typ`: Global layout definitions for different document types.

## Color Palettes (`color.typ`)

We maintain several color schemes tailored for different contexts. All colors are defined as Typst `rgb` constants.

### 1. Base Colors

The foundational palette for standard UI elements and text.

- **Primary BG**: `#dde3f0`
- **Primary Text**: `#1f1804`

### 2. Specialized Themes

| Theme Name       | Mode     | Inspiration / Usage                                    |
| :--------------- | :------- | :----------------------------------------------------- |
| `desert-rose`    | Light    | Earthy tones, warm aesthetics.                         |
| `faded-glory`    | Light    | Sophisticated grays and muted blues.                   |
| `indigo-languor` | Light    | High-contrast blue and professional whites.            |
| `forest`         | Light    | Nature-inspired greens and teals.                      |
| `hintze-hall`    | **Dark** | High-end dark mode inspired by architectural sketches. |

## Document Types

Our documentation is categorized into four main formats. When creating a new file, please follow the naming convention: `{type}_{topic}.typ`.

- **PAPER**: Formal research, whitepapers, or academic-style reports.
- **PROGRESS**: Periodic status updates and milestone tracking.
- **NOTE**: Quick technical notes, brain-dumping, and internal references.
- **LOG**: Changelogs, meeting minutes, and daily activity records.

## Usage in Typst

Refer to [simple-research-poster](https://typst.app/universe/package/simple-research-poster).

To use these colors in your `.typ` files, import the library as follows:

```typst
#import "@preview/simple-research-poster:0.1.0": *
#import "{path to lib}/color.typ": *
#import "{path to lib}/tag.typ": *

#show: poster.with(
  title:       [Applied Cryonics: How I Slept Through the 21st Century],
  author:      [Philip J. Fry and Turanga Leela],
  mentor:      [Mentor: Bender B. Rodriguez],
  width:       43in,
  height:      32.5in,
  subtitle:    [College of New New York],
  logo:        image("assets/logo.png", height: 150%),
  base-colors: base-colors
)
#let colored-poster-section = poster-section.with(base-colors: base-colors)

#show: tag.with(
  "NOTE",
  bg_color: indigo-languor.highlight,
  text_color: indigo-languor.cover2
)

#pad(
  grid(
    columns: 3,
    inset: 0.5in,
    gutter: 30pt,
    [
      #colored-poster-section[Section 1][#section1]
      #colored-poster-section(fill: true)[Section 2][#section2]
    ],
    [
      #colored-poster-section[Section 3][#section3]
      #colored-poster-section(fill: true)[Section 4][#section4]
      #colored-poster-section[Section 5][#section5]
    ],
    [
      #colored-poster-section(fill: true)[Section 6][#section6]
      #colored-poster-section[Section 7][#section7]
      #colored-poster-section(fill: true)[Acknowledgements][#acknowledgements]
      #colored-poster-section[References][#references]
    ]
  ),
  top: 0.5in,
  x: 1in,
)
```
