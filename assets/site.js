"use strict";

// Progressive enhancement: without JS, every task and the full citation remain visible.
const tabs = [...document.querySelectorAll('[role="tab"]')];
const panels = [...document.querySelectorAll(".query-panel")];
if (tabs.length && panels.length === tabs.length) {
  const selectTab = (index, focus = false) => {
    tabs.forEach((tab, i) => {
      const active = i === index;
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
      panels[i].hidden = !active;
    });
    if (focus) tabs[index].focus();
  };
  tabs.forEach((tab, index) => {
    panels[index].setAttribute("role", "tabpanel");
    panels[index].setAttribute("aria-labelledby", tab.id);
    panels[index].tabIndex = 0;
    tab.addEventListener("click", () => selectTab(index));
    tab.addEventListener("keydown", (event) => {
      let next = index;
      if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
      else if (event.key === "ArrowLeft")
        next = (index + tabs.length - 1) % tabs.length;
      else if (event.key === "Home") next = 0;
      else if (event.key === "End") next = tabs.length - 1;
      else return;
      event.preventDefault();
      selectTab(next, true);
    });
  });
  selectTab(0);
  document.querySelector(".query-tabs").hidden = false;
  document.querySelector(".query-explorer").classList.add("is-interactive");
}

const copyButton = document.querySelector("[data-copy-citation]");
const citation = document.getElementById("bibtex");
const copyStatus = document.getElementById("copy-status");
if (copyButton && citation && copyStatus) {
  copyButton.hidden = false;
  copyButton.addEventListener("click", async () => {
    try {
      if (!navigator.clipboard?.writeText)
        throw new Error("Clipboard API unavailable");
      await navigator.clipboard.writeText(citation.textContent.trim() + "\n");
      copyStatus.textContent = "Citation copied to clipboard.";
      copyButton.textContent = "Copied ✓";
      setTimeout(() => {
        copyButton.textContent = "Copy citation";
      }, 2500);
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(citation);
      selection.removeAllRanges();
      selection.addRange(range);
      copyStatus.textContent =
        "Automatic copy is unavailable. The citation is selected for manual copying.";
    }
  });
}

const dialog = document.getElementById("figure-dialog");
if (dialog && typeof dialog.showModal === "function") {
  const image = document.getElementById("figure-dialog-image");
  const title = document.getElementById("figure-dialog-title");
  document.querySelectorAll("[data-figure]").forEach((link) => {
    link.addEventListener("click", (event) => {
      // Preserve browser modifiers such as Cmd/Ctrl-click to open the original image.
      if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey)
        return;
      event.preventDefault();
      image.src = link.href;
      image.alt = link.querySelector("img").alt;
      title.textContent = link.dataset.caption;
      dialog.showModal();
      dialog.querySelector(".dialog-image-scroll").scrollTo(0, 0);
    });
  });
  dialog
    .querySelector("[data-close-figure]")
    .addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (event) => {
    if (event.target !== dialog) return;
    const bounds = dialog.getBoundingClientRect();
    if (
      event.clientX < bounds.left ||
      event.clientX > bounds.right ||
      event.clientY < bounds.top ||
      event.clientY > bounds.bottom
    )
      dialog.close();
  });
}

if ("IntersectionObserver" in window) {
  const links = [...document.querySelectorAll(".nav-links a")];
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        links.forEach((link) => {
          if (link.hash === `#${entry.target.id}`)
            link.setAttribute("aria-current", "location");
          else link.removeAttribute("aria-current");
        });
      });
    },
    { rootMargin: "-15% 0px -65% 0px", threshold: 0 },
  );
  links.forEach((link) => {
    const section = document.querySelector(link.hash);
    if (section) observer.observe(section);
  });
  const hero = document.getElementById("top");
  if (hero) observer.observe(hero);
}
