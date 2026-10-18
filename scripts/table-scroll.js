/* Keep native table semantics; only the surrounding region scrolls. */
(() => {
  function wrapTables() {
    document.querySelectorAll('.post-content table').forEach((table, index) => {
      if (table.closest('.table-scroll')) return;
      const wrapper = document.createElement('div');
      wrapper.className = 'table-scroll';
      wrapper.tabIndex = 0;
      wrapper.setAttribute('role', 'region');
      const caption = table.querySelector('caption')?.textContent.trim();
      wrapper.setAttribute('aria-label', caption ? `Table: ${caption}` : `Data table ${index + 1}; scroll horizontally if needed`);
      table.before(wrapper);
      wrapper.append(table);
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', wrapTables, {once: true});
  else wrapTables();
})();
