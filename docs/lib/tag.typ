#let tag(tag_name, bg_color: blue, text_color: black, body) = {
  body

  place(
    top + left,
    dy: -4.2in,
    dx: 1in,
    rect(width: 4.2in, height: 2.6in, fill: bg_color)[
      #place(
        bottom + right,
        text(3em, fill: text_color)[#tag_name],
      )
    ],
  )
}

// #show: tag.with(
//   "NOTE",
//   color: rgb("#838c83"),
// )
