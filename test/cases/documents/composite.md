---
title: "A Complex HTML + Markdown Mashup"
author: Jane Doe
tags: [markdown, html, demo]
---

<!-- This is an HTML comment inside a Markdown document. -->

# A Complex HTML + Markdown Mashup

Welcome to **Markdown** with <em>HTML</em>, <strong>both</strong>, and even <mark>highlighted text</mark>.  
Use <kbd>Ctrl</kbd> + <kbd>C</kbd> to copy, and press <kbd>⌘</kbd> + <kbd>V</kbd> on macOS.  
The chemical formula for water is H<sub>2</sub>O, and Einstein’s famous equation is E = mc<sup>2</sup>.

> **Blockquote in Markdown**
>
> This is still Markdown, but it contains <abbr title="HyperText Markup Language">HTML</abbr>.
>
> <blockquote>
>   <p>And this is an HTML blockquote nested inside a Markdown blockquote.</p>
> </blockquote>

---

## 1. Lists, Tasks, and HTML

- [x] Write the **Markdown** draft
- [ ] Embed <code>&lt;html&gt;</code> tags
- [ ] Validate with <a href="https://commonmark.org/" target="_blank">CommonMark</a>

<div class="callout" style="border-left: 4px solid #f90; padding: 0.75rem 1rem; background: #fff8e6;">
  <p><strong>Note:</strong> This entire <code>&lt;div&gt;</code> is raw HTML. Some Markdown parsers will not process Markdown inside it unless you add blank lines or enable a special mode.</p>
</div>

### Nested Markdown list with inline HTML

1. First item
   - Sub-item with `inline code`
   - Sub-item with <span style="color: crimson;">a red HTML span</span>
2. Second item
   - [Link via Markdown](https://example.com)
   - <a href="https://example.org">Link via HTML</a>

---

## 2. Table: Markdown vs. HTML

| Feature | Markdown | HTML |
| :--- | :---: | ---: |
| Bold | `**text**` | `<strong>text</strong>` |
| Italic | `_text_` | `<em>text</em>` |
| Line break | two spaces or `\` | `<br>` |
| Image | `![alt](url)` | `<img src="url" alt="alt">` |

<table>
  <caption>An HTML table with Markdown-ish content</caption>
  <thead>
    <tr>
      <th>Element</th>
      <th>Example</th>
      <th>Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>&lt;details&gt;</code></td>
      <td>
        <details>
          <summary>Click to expand</summary>
          <p>Hidden content with <strong>HTML bold</strong> and <a href="https://example.com">a link</a>.</p>
        </details>
      </td>
      <td>Good for FAQs.</td>
    </tr>
    <tr>
      <td><code>&lt;figure&gt;</code></td>
      <td>
        <figure>
          <img src="https://via.placeholder.com/120x60?text=HTML" alt="Placeholder">
          <figcaption>An HTML figure caption.</figcaption>
        </figure>
      </td>
      <td>Semantic media.</td>
    </tr>
  </tbody>
</table>

---

## 3. Code, Quotes, and Footnotes

Here is a Python snippet:

```python
def greet(name: str) -> str:
    return f"Hello, {name}!"

print(greet("Markdown + HTML"))
```

And here is an HTML snippet:

```html
<section class="hero">
  <h1>Hello, <span class="accent">world</span>!</h1>
  <p>This is <strong>HTML</strong> inside a Markdown code fence.</p>
</section>
```

> “Markdown is intended to be as easy-to-read and easy-to-write as is feasible.”
>
> — <cite>John Gruber</cite>

A footnote reference looks like this.[^1]

[^1]: This is the footnote text. It may contain <em>HTML</em>, `code`, and [links](https://example.com).

---

## 4. Definition List and Details

<dl>
  <dt>Markdown</dt>
  <dd>A lightweight markup language with plain-text formatting syntax.</dd>
  <dt>HTML</dt>
  <dd>The standard language for documents designed to be displayed in a web browser.</dd>
</dl>

<details open>
  <summary><strong>Open by default</strong> (HTML <code>&lt;details open&gt;</code>)</summary>

  This paragraph is Markdown inside an HTML `<details>` block.  
  It contains **bold**, _italic_, and a [link](https://example.com).

  - Item one
  - Item two

</details>

---

## 5. Mixed Inline Madness

This sentence has **Markdown bold**, <strong>HTML bold</strong>, _Markdown italic_, <em>HTML italic</em>, ~~strikethrough~~, <del>HTML deleted text</del>, `inline code`, <code>HTML code</code>, and a footnote.[^2]  
It also has an image: ![Markdown image](https://via.placeholder.com/80x40?text=MD) and an HTML image: <img src="https://via.placeholder.com/80x40?text=HTML" alt="HTML image" width="80" height="40">.

<aside style="border: 1px dashed #999; padding: 1rem;">
  <h3>An HTML <code>&lt;aside&gt;</code></h3>
  <p>This is outside the normal Markdown flow, but it can still contain <a href="https://example.com">links</a>, <strong>bold</strong>, and <code>code</code>.</p>
</aside>

[^2]: Another footnote with <mark>highlighted HTML</mark>.

---

<p align="center">
  <small>&copy; 2025 Example Corp. Built with <strong>Markdown</strong> + <em>HTML</em>.</small>
</p>
