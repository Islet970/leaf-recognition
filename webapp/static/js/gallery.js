/* ============================================================
   校园叶片识别与可解释性可视化系统 — pic 数据看板
   原生 JavaScript ES6 + 本地 Apache ECharts 5.6
   ============================================================ */

(function () {
  'use strict';

  var PALETTE = ['#3d8b5c', '#7fb98d', '#b7a34a', '#5b8db8', '#a4626b', '#6a8f5f', '#c07f4a', '#8b7bb0'];
  var MAX_TABLE_ROWS = 20;

  var D = {};

  var state = {
    loaded: false,
    loading: false,
    pendingRender: false,
    pendingTarget: 0,
    directories: [],
    activeIndex: -1,
    charts: [],
    lightbox: {
      images: [],
      index: 0,
      trigger: null
    }
  };

  document.addEventListener('DOMContentLoaded', initGallery);

  function initGallery() {
    cacheDom();
    bindControls();

    document.addEventListener('app:sectionchange', function (event) {
      if (event.detail && event.detail.section === 'gallery') {
        ensureLoaded(false);
        window.requestAnimationFrame(resizeAllCharts);
      }
    });

    /* 启动时先预取 /api/gallery；图表等到首次进入看板时再初始化，避免隐藏容器尺寸为 0 */
    ensureLoaded(false);
  }

  function cacheDom() {
    D.section = document.getElementById('section-gallery');
    D.status = document.getElementById('galleryStatus');
    D.tabs = document.getElementById('galleryTabs');
    D.panel = document.getElementById('galleryPanel');
    D.refresh = document.getElementById('galleryRefresh');

    D.lightbox = document.getElementById('lightbox');
    D.lightboxImage = document.getElementById('lightboxImage');
    D.lightboxCaption = document.getElementById('lightboxCaption');
    D.lightboxDownload = document.getElementById('lightboxDownload');
    D.lightboxClose = document.getElementById('lightboxClose');
    D.lightboxPrev = document.getElementById('lightboxPrev');
    D.lightboxNext = document.getElementById('lightboxNext');
  }

  /* ============================================================
     加载与状态
     ============================================================ */

  function ensureLoaded(force) {
    if (state.loading) {
      return;
    }
    if (state.loaded && !force) {
      if (state.pendingRender) {
        state.pendingRender = false;
        selectDirectory(state.pendingTarget, true);
        return;
      }
      window.requestAnimationFrame(resizeAllCharts);
      return;
    }

    state.loading = true;
    if (D.refresh) {
      D.refresh.disabled = true;
    }
    showStatus('正在加载数据看板…', false);

    var previousName = '';
    if (state.directories[state.activeIndex]) {
      previousName = state.directories[state.activeIndex].name;
    }

    fetchJSON('/api/gallery').then(function (data) {
      state.loading = false;
      if (D.refresh) {
        D.refresh.disabled = false;
      }

      state.directories = Array.isArray(data.directories) ? data.directories : [];
      state.loaded = true;

      if (!state.directories.length) {
        hideStatus();
        renderTabs();
        showEmptyPanel('pic/ 下暂无数据目录');
        return;
      }

      hideStatus();
      renderTabs();

      var target = 0;
      var found = false;

      if (previousName) {
        state.directories.forEach(function (dir, index) {
          if (dir.name === previousName) {
            target = index;
            found = true;
          }
        });
      }

      if (!found) {
        state.directories.forEach(function (dir, index) {
          if (!found && visibleFiles(dir).length) {
            target = index;
            found = true;
          }
        });
      }

      if (D.section && D.section.hidden) {
        state.pendingTarget = target;
        state.pendingRender = true;
        return;
      }

      selectDirectory(target, true);
    }).catch(function (err) {
      state.loading = false;
      if (D.refresh) {
        D.refresh.disabled = false;
      }
      showStatus('数据看板加载失败：' + err.message + '。点击「刷新数据」重试。', true);
      toast('数据看板加载失败：' + err.message, 'error', 5600);
    });
  }

  function bindControls() {
    if (D.refresh) {
      D.refresh.addEventListener('click', function () {
        ensureLoaded(true);
      });
    }

    if (D.lightboxClose) {
      D.lightboxClose.addEventListener('click', closeLightbox);
      D.lightboxPrev.addEventListener('click', function () {
        stepLightbox(-1);
      });
      D.lightboxNext.addEventListener('click', function () {
        stepLightbox(1);
      });
      D.lightbox.addEventListener('click', function (event) {
        if (event.target === D.lightbox) {
          closeLightbox();
        }
      });
      document.addEventListener('keydown', handleLightboxKeys);
    }
  }

  function showStatus(text, isError) {
    if (!D.status) {
      return;
    }
    D.status.hidden = false;
    D.status.textContent = text;
    D.status.classList.toggle('is-error', Boolean(isError));
  }

  function hideStatus() {
    if (D.status) {
      D.status.hidden = true;
    }
  }

  /* ============================================================
     目录 Tab
     ============================================================ */

  function renderTabs() {
    if (!D.tabs) {
      return;
    }
    D.tabs.textContent = '';

    if (!state.directories.length) {
      D.tabs.hidden = true;
      return;
    }
    D.tabs.hidden = false;

    var buttons = [];

    state.directories.forEach(function (dir, index) {
      var files = visibleFiles(dir);
      var button = document.createElement('button');
      button.type = 'button';
      button.className = 'gallery-tab';
      button.id = 'gallery-tab-' + index;
      button.setAttribute('role', 'tab');
      button.setAttribute('aria-selected', index === state.activeIndex ? 'true' : 'false');
      button.tabIndex = index === state.activeIndex ? 0 : -1;
      button.setAttribute('data-index', String(index));

      var label = document.createElement('span');
      label.textContent = dir.name || '(未命名目录)';
      var count = document.createElement('span');
      count.className = 'gallery-tab-count';
      count.textContent = String(files.length);

      button.appendChild(label);
      button.appendChild(count);
      button.addEventListener('click', function () {
        selectDirectory(index);
      });

      D.tabs.appendChild(button);
      buttons.push(button);
    });

    Array.prototype.forEach.call(buttons, function (button, index) {
      button.addEventListener('keydown', function (event) {
        var last = buttons.length - 1;
        var next = -1;

        if (event.key === 'ArrowRight') {
          next = index === last ? 0 : index + 1;
        } else if (event.key === 'ArrowLeft') {
          next = index === 0 ? last : index - 1;
        } else if (event.key === 'Home') {
          next = 0;
        } else if (event.key === 'End') {
          next = last;
        }

        if (next >= 0) {
          event.preventDefault();
          buttons[next].focus();
          selectDirectory(next);
        }
      });
    });
  }

  function selectDirectory(index, silent) {
    if (index < 0 || index >= state.directories.length) {
      return;
    }

    disposeCharts();
    state.activeIndex = index;
    state.pendingRender = false;

    if (D.panel) {
      D.panel.setAttribute('aria-labelledby', 'gallery-tab-' + index);
    }

    if (D.tabs) {
      var tabs = D.tabs.querySelectorAll('.gallery-tab');
      Array.prototype.forEach.call(tabs, function (tab, tabIndex) {
        var selected = tabIndex === index;
        tab.setAttribute('aria-selected', selected ? 'true' : 'false');
        tab.classList.toggle('is-active', selected);
        tab.tabIndex = selected ? 0 : -1;
      });
    }

    renderDirectory(state.directories[index]);
    window.requestAnimationFrame(resizeAllCharts);

    if (!silent && D.panel) {
      D.panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  /* ============================================================
     目录内容
     ============================================================ */

  function renderDirectory(dir) {
    if (!D.panel) {
      return;
    }
    D.panel.textContent = '';

    var files = visibleFiles(dir).slice();
    files.sort(function (a, b) {
      return compareNames(a.name, b.name);
    });

    if (!files.length) {
      showEmptyPanel('「' + dir.name + '」暂无数据');
      return;
    }

    var csvFiles = [];
    var imageFiles = [];
    var docFiles = [];

    files.forEach(function (file) {
      var kind = fileKind(file);
      if (kind === 'csv') {
        csvFiles.push(file);
      } else if (kind === 'image') {
        imageFiles.push(file);
      } else if (kind === 'doc') {
        docFiles.push(file);
      }
    });

    csvFiles.forEach(function (file) {
      renderCsv(dir, file);
    });

    if (imageFiles.length) {
      renderImageSection(dir, imageFiles);
    }

    if (docFiles.length) {
      renderDocSection(dir, docFiles);
    }

    if (!csvFiles.length && !imageFiles.length && !docFiles.length) {
      showEmptyPanel('「' + dir.name + '」暂无可展示的文件');
    }
  }

  function showEmptyPanel(text) {
    if (!D.panel) {
      return;
    }
    D.panel.textContent = '';
    var box = document.createElement('div');
    box.className = 'loading-state is-empty';
    box.textContent = text;
    D.panel.appendChild(box);
  }

  /* ============================================================
     CSV
     ============================================================ */

  function renderCsv(dir, file) {
    var parseFailed = Boolean(file.error) ||
      (String(file.type || '').toLowerCase() === 'file' && fileExt(file.name) === 'csv');

    var card = document.createElement('section');
    card.className = 'csv-card';

    var title = document.createElement('h4');
    title.textContent = file.name;
    var meta = document.createElement('span');
    meta.className = 'csv-meta';
    meta.textContent = parseFailed
      ? '解析失败'
      : '预览 ' + (Array.isArray(file.rows) ? file.rows.length : 0) +
        ' 行 / 共 ' + (Number(file.n_rows) || 0) + ' 行';
    title.appendChild(meta);
    card.appendChild(title);

    if (parseFailed) {
      var errorText = document.createElement('p');
      errorText.className = 'table-note is-error';
      var reason = file.error || '后端未能解析该文件';
      errorText.textContent = reason.indexOf('CSV 解析失败') === 0
        ? reason
        : 'CSV 解析失败：' + reason;

      var download = document.createElement('a');
      download.className = 'btn btn-small csv-download';
      download.href = fileUrl(dir, file);
      download.setAttribute('download', file.name);
      download.textContent = '下载原始 CSV';

      card.appendChild(errorText);
      card.appendChild(download);
      D.panel.appendChild(card);
      return;
    }

    var csv = normalizeCsv(file);
    var kind = csvKind(csv.columns);

    if (!csv.rows.length) {
      var empty = document.createElement('p');
      empty.className = 'table-note';
      empty.textContent = 'CSV 暂无数据';
      card.appendChild(empty);
      D.panel.appendChild(card);
      return;
    }

    D.panel.appendChild(card);

    if (kind === 'metrics') {
      renderMetricsCharts(card, csv);
    } else if (kind === 'ablation') {
      renderAblationCharts(card, csv);
    } else {
      renderTable(card, csv);
    }
  }

  function normalizeCsv(file) {
    var columns = Array.isArray(file.columns) ? file.columns.map(function (column) {
      return String(column).replace(/^\uFEFF/, '').trim();
    }) : [];
    var rows = Array.isArray(file.rows) ? file.rows : [];

    if (!columns.length && rows.length) {
      var first = rowObject(rows[0], []);
      columns = Object.keys(first);
    }

    var total = Number(file.n_rows);
    if (!isFinite(total) || total < rows.length) {
      total = rows.length;
    }

    return {
      columns: columns,
      rows: rows,
      nRows: total
    };
  }

  function rowObject(row, columns) {
    if (Array.isArray(row)) {
      var mapped = {};
      columns.forEach(function (column, index) {
        mapped[column] = row[index];
      });
      return mapped;
    }
    return row && typeof row === 'object' ? row : {};
  }

  function rowLookup(row, columns) {
    var source = rowObject(row, columns);
    var normalized = {};
    Object.keys(source).forEach(function (key) {
      normalized[String(key).replace(/^\uFEFF/, '').trim().toLowerCase()] = source[key];
    });
    return normalized;
  }

  function csvKind(columns) {
    var lower = columns.map(function (column) {
      return String(column).trim().toLowerCase();
    });
    var has = function (name) {
      return lower.indexOf(name) !== -1;
    };

    if (has('dataset') && has('model') && has('accuracy') && has('ce_loss')) {
      return 'ablation';
    }
    if (has('model') && has('accuracy') && has('ce_loss')) {
      return 'metrics';
    }
    return 'table';
  }

  function renderMetricsCharts(card, csv) {
    var labels = [];
    var accuracy = [];
    var loss = [];

    csv.rows.forEach(function (row, index) {
      var item = rowLookup(row, csv.columns);
      labels.push(item.model === undefined || item.model === '' ? '#' + (index + 1) : String(item.model));
      accuracy.push(toNumber(item.accuracy));
      loss.push(toNumber(item.ce_loss));
    });

    var grid = document.createElement('div');
    grid.className = 'chart-grid';

    var accuracyChart = createChartCard('测试准确率 accuracy');
    var lossChart = createChartCard('测试交叉熵 ce_loss');
    grid.appendChild(accuracyChart.card);
    grid.appendChild(lossChart.card);
    card.appendChild(grid);

    initChart(accuracyChart.chart, barOption(labels, accuracy, {
      unit: 'percent',
      color: '#3d8b5c',
      max: 1
    }));
    initChart(lossChart.chart, barOption(labels, loss, {
      unit: 'number',
      color: '#b7a34a'
    }));
  }

  function renderAblationCharts(card, csv) {
    var datasets = [];
    var models = [];
    var lookup = {};

    csv.rows.forEach(function (row) {
      var item = rowLookup(row, csv.columns);
      var dataset = item.dataset === undefined ? '' : String(item.dataset);
      var model = item.model === undefined ? '' : String(item.model);
      if (!dataset || !model) {
        return;
      }
      if (datasets.indexOf(dataset) === -1) {
        datasets.push(dataset);
      }
      if (models.indexOf(model) === -1) {
        models.push(model);
      }
      lookup[dataset + '||' + model] = item;
    });

    if (!datasets.length || !models.length) {
      var empty = document.createElement('p');
      empty.className = 'table-note';
      empty.textContent = 'CSV 缺少 dataset / model 列，无法绘制分组柱状图';
      card.appendChild(empty);
      return;
    }

    var grid = document.createElement('div');
    grid.className = 'chart-grid';

    var accuracyChart = createChartCard('各数据集准确率 accuracy');
    var lossChart = createChartCard('各数据集交叉熵 ce_loss');
    grid.appendChild(accuracyChart.card);
    grid.appendChild(lossChart.card);
    card.appendChild(grid);

    initChart(accuracyChart.chart, groupedBarOption(datasets, models, lookup, {
      metric: 'accuracy',
      unit: 'percent',
      max: 1
    }));
    initChart(lossChart.chart, groupedBarOption(datasets, models, lookup, {
      metric: 'ce_loss',
      unit: 'number'
    }));
  }

  function createChartCard(titleText) {
    var card = document.createElement('div');
    card.className = 'chart-card';

    var title = document.createElement('h4');
    title.textContent = titleText;

    var chart = document.createElement('div');
    chart.className = 'chart';

    card.appendChild(title);
    card.appendChild(chart);

    return { card: card, chart: chart };
  }

  function initChart(chartElement, option) {
    if (!window.echarts) {
      chartElement.className = 'chart-empty';
      chartElement.textContent = 'ECharts 未加载';
      return;
    }
    try {
      var chart = window.echarts.init(chartElement, null, { renderer: 'canvas' });
      chart.setOption(option);
      state.charts.push(chart);
    } catch (err) {
      chartElement.className = 'chart-empty';
      chartElement.textContent = '图表渲染失败：' + err.message;
    }
  }

  function baseOption() {
    return {
      textStyle: {
        fontFamily: '"Microsoft YaHei", "PingFang SC", sans-serif',
        color: '#3c4f44'
      },
      animationDuration: 420
    };
  }

  function barOption(labels, values, config) {
    var option = baseOption();
    option.color = [config.color || PALETTE[0]];
    option.grid = { left: 48, right: 22, top: 34, bottom: labels.length > 4 ? 52 : 40 };
    option.tooltip = {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      valueFormatter: function (value) {
        return formatMetric(value, config.unit);
      }
    };
    option.xAxis = {
      type: 'category',
      data: labels,
      axisTick: { alignWithLabel: true },
      axisLine: { lineStyle: { color: '#c3d8c9' } },
      axisLabel: {
        color: '#64776c',
        interval: 0,
        rotate: labels.some(function (label) { return String(label).length > 8; }) ? 18 : 0
      }
    };
    option.yAxis = {
      type: 'value',
      min: 0,
      max: config.max,
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: {
        color: '#64776c',
        formatter: config.unit === 'percent'
          ? function (value) { return Math.round(value * 100) + '%'; }
          : undefined
      },
      splitLine: { lineStyle: { color: '#eef4ef' } }
    };
    option.series = [{
      name: config.unit === 'percent' ? 'accuracy' : 'ce_loss',
      type: 'bar',
      data: values,
      barMaxWidth: 44,
      itemStyle: {
        borderRadius: [6, 6, 0, 0],
        color: config.color || PALETTE[0]
      },
      label: {
        show: true,
        position: 'top',
        color: '#2f6b45',
        fontSize: 10,
        formatter: function (params) {
          return formatMetric(params.value, config.unit);
        }
      }
    }];

    return option;
  }

  function groupedBarOption(datasets, models, lookup, config) {
    var option = baseOption();
    option.color = PALETTE;
    option.grid = { left: 48, right: 24, top: 52, bottom: 54 };
    option.legend = {
      top: 2,
      type: 'scroll',
      textStyle: { color: '#3c4f44', fontSize: 11 }
    };
    option.tooltip = {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      valueFormatter: function (value) {
        return formatMetric(value, config.unit);
      }
    };
    option.xAxis = {
      type: 'category',
      data: datasets,
      axisTick: { alignWithLabel: true },
      axisLine: { lineStyle: { color: '#c3d8c9' } },
      axisLabel: {
        color: '#64776c',
        interval: 0,
        rotate: datasets.some(function (label) { return String(label).length > 8; }) ? 16 : 0
      }
    };
    option.yAxis = {
      type: 'value',
      min: 0,
      max: config.max,
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: {
        color: '#64776c',
        formatter: config.unit === 'percent'
          ? function (value) { return Math.round(value * 100) + '%'; }
          : undefined
      },
      splitLine: { lineStyle: { color: '#eef4ef' } }
    };

    option.series = models.map(function (model) {
      var values = datasets.map(function (dataset) {
        var row = lookup[dataset + '||' + model];
        return row ? toNumber(row[config.metric]) : null;
      });
      return {
        name: model,
        type: 'bar',
        data: values,
        barMaxWidth: 26,
        itemStyle: { borderRadius: [4, 4, 0, 0] },
        label: {
          show: models.length <= 4,
          position: 'top',
          fontSize: 9,
          color: '#3c4f44',
          formatter: function (params) {
            return formatMetric(params.value, config.unit);
          }
        }
      };
    });

    return option;
  }

  function renderTable(card, csv) {
    var wrapper = document.createElement('div');
    wrapper.className = 'table-wrap';

    var table = document.createElement('table');
    table.className = 'data-table';

    var thead = document.createElement('thead');
    var headRow = document.createElement('tr');
    csv.columns.forEach(function (column) {
      var th = document.createElement('th');
      th.scope = 'col';
      th.textContent = column;
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    var tbody = document.createElement('tbody');
    var preview = csv.rows.slice(0, MAX_TABLE_ROWS);
    preview.forEach(function (row) {
      var item = rowObject(row, csv.columns);
      var tr = document.createElement('tr');
      csv.columns.forEach(function (column) {
        var td = document.createElement('td');
        var value = item[column];
        td.textContent = value === undefined || value === null ? '' : String(value);
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrapper.appendChild(table);
    card.appendChild(wrapper);

    var note = document.createElement('p');
    note.className = 'table-note';
    note.textContent = '表格预览前 ' + preview.length + ' 行，共 ' + csv.nRows + ' 行';
    card.appendChild(note);
  }

  function toNumber(value) {
    if (value === null || value === undefined || value === '') {
      return null;
    }
    var number = Number(value);
    return isFinite(number) ? number : null;
  }

  function formatMetric(value, unit) {
    var number = toNumber(value);
    if (number === null) {
      return '—';
    }
    if (unit === 'percent') {
      return (number * 100).toFixed(1) + '%';
    }
    return number.toFixed(3);
  }

  /* ============================================================
     图片与文档
     ============================================================ */

  function renderImageSection(dir, images) {
    var section = document.createElement('section');
    section.className = 'gallery-section';

    var head = document.createElement('div');
    head.className = 'gallery-section-head';
    var title = document.createElement('h3');
    title.textContent = '图片';
    var count = document.createElement('span');
    count.textContent = images.length + ' 张 · 点击查看大图';
    head.appendChild(title);
    head.appendChild(count);
    section.appendChild(head);

    var grid = document.createElement('div');
    grid.className = 'image-grid';

    var entries = images.map(function (file) {
      return {
        name: file.name,
        url: fileUrl(dir, file),
        dir: dir.name
      };
    });

    images.forEach(function (file, index) {
      var button = document.createElement('button');
      button.type = 'button';
      button.className = 'thumb';
      button.title = file.name;

      var img = document.createElement('img');
      img.src = fileUrl(dir, file);
      img.alt = file.name;
      img.loading = 'lazy';
      img.decoding = 'async';

      var caption = document.createElement('span');
      caption.className = 'thumb-caption';
      var name = document.createElement('strong');
      name.textContent = file.name;
      var meta = document.createElement('small');
      meta.textContent = formatBytes(file.size) + (file.mtime ? ' · ' + formatDate(file.mtime) : '');
      caption.appendChild(name);
      caption.appendChild(meta);

      button.appendChild(img);
      button.appendChild(caption);
      button.addEventListener('click', function () {
        openLightbox(entries, index, button);
      });

      grid.appendChild(button);
    });

    section.appendChild(grid);
    D.panel.appendChild(section);
  }

  function renderDocSection(dir, docs) {
    var section = document.createElement('section');
    section.className = 'gallery-section';

    var head = document.createElement('div');
    head.className = 'gallery-section-head';
    var title = document.createElement('h3');
    title.textContent = '文档下载';
    var count = document.createElement('span');
    count.textContent = docs.length + ' 个文件';
    head.appendChild(title);
    head.appendChild(count);
    section.appendChild(head);

    var list = document.createElement('div');
    list.className = 'doc-list';

    docs.forEach(function (file) {
      var link = document.createElement('a');
      link.className = 'doc-link';
      link.href = fileUrl(dir, file);
      link.setAttribute('download', file.name);

      var icon = document.createElement('span');
      icon.className = 'doc-icon';
      icon.setAttribute('aria-hidden', 'true');
      icon.textContent = fileExt(file.name).toUpperCase();

      var meta = document.createElement('span');
      meta.className = 'doc-meta';
      var name = document.createElement('strong');
      name.textContent = file.name;
      var info = document.createElement('small');
      info.textContent = formatBytes(file.size) + (file.mtime ? ' · ' + formatDate(file.mtime) : '');
      meta.appendChild(name);
      meta.appendChild(info);

      link.appendChild(icon);
      link.appendChild(meta);
      list.appendChild(link);
    });

    section.appendChild(list);
    D.panel.appendChild(section);
  }

  /* ============================================================
     灯箱
     ============================================================ */

  function openLightbox(images, index, trigger) {
    state.lightbox.images = images;
    state.lightbox.index = index;
    state.lightbox.trigger = trigger || null;

    renderLightbox();
    D.lightbox.hidden = false;
    document.body.classList.add('is-locked');

    window.setTimeout(function () {
      D.lightboxClose.focus();
    }, 0);
  }

  function renderLightbox() {
    var images = state.lightbox.images;
    if (!images.length) {
      return;
    }
    var index = state.lightbox.index;
    var item = images[index];

    D.lightboxImage.src = item.url;
    D.lightboxImage.alt = item.dir + ' / ' + item.name;
    D.lightboxCaption.textContent = item.dir + ' / ' + item.name;
    D.lightboxDownload.href = item.url;
    D.lightboxDownload.setAttribute('download', item.name);

    var single = images.length <= 1;
    D.lightboxPrev.disabled = single;
    D.lightboxNext.disabled = single;
  }

  function stepLightbox(delta) {
    var images = state.lightbox.images;
    if (images.length <= 1) {
      return;
    }
    state.lightbox.index = (state.lightbox.index + delta + images.length) % images.length;
    renderLightbox();
  }

  function closeLightbox() {
    if (!D.lightbox || D.lightbox.hidden) {
      return;
    }
    D.lightbox.hidden = true;
    D.lightboxImage.removeAttribute('src');
    document.body.classList.remove('is-locked');

    if (state.lightbox.trigger && typeof state.lightbox.trigger.focus === 'function') {
      state.lightbox.trigger.focus();
    }
    state.lightbox.trigger = null;
  }

  function handleLightboxKeys(event) {
    if (!D.lightbox || D.lightbox.hidden) {
      return;
    }
    if (event.key === 'Escape') {
      event.preventDefault();
      closeLightbox();
    } else if (event.key === 'ArrowLeft') {
      event.preventDefault();
      stepLightbox(-1);
    } else if (event.key === 'ArrowRight') {
      event.preventDefault();
      stepLightbox(1);
    } else if (event.key === 'Tab') {
      trapLightboxFocus(event);
    }
  }

  function trapLightboxFocus(event) {
    var focusable = D.lightbox.querySelectorAll('button:not([disabled]), [href]');
    if (!focusable.length) {
      return;
    }
    var first = focusable[0];
    var last = focusable[focusable.length - 1];

    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  /* ============================================================
     工具
     ============================================================ */

  function visibleFiles(dir) {
    var files = Array.isArray(dir.files) ? dir.files : [];
    return files.filter(function (file) {
      if (!file || !file.name) {
        return false;
      }
      var name = String(file.name);
      return name.indexOf('~$') !== 0 && name.charAt(0) !== '.';
    });
  }

  function fileKind(file) {
    var type = String(file.type || '').toLowerCase();
    var ext = fileExt(file.name);

    if (type === 'csv' || ext === 'csv') {
      return 'csv';
    }
    if (type === 'image' || type === 'png' || type === 'jpg' || type === 'jpeg' ||
        ext === 'png' || ext === 'jpg' || ext === 'jpeg') {
      return 'image';
    }
    if (type === 'docx' || type === 'pdf' || ext === 'docx' || ext === 'pdf') {
      return 'doc';
    }
    return 'other';
  }

  function fileExt(name) {
    var text = String(name || '');
    var dot = text.lastIndexOf('.');
    return dot === -1 ? '' : text.slice(dot + 1).toLowerCase();
  }

  function fileUrl(dir, file) {
    if (file.url) {
      return file.url;
    }
    return '/pic/' + encodeURIComponent(dir.name) + '/' + encodeURIComponent(file.name);
  }

  function compareNames(a, b) {
    return String(a).localeCompare(String(b), 'zh-CN', {
      numeric: true,
      sensitivity: 'base'
    });
  }

  function formatBytes(bytes) {
    var size = Number(bytes);
    if (!isFinite(size) || size <= 0) {
      return '0 B';
    }
    if (size < 1024) {
      return size + ' B';
    }
    if (size < 1024 * 1024) {
      return (size / 1024).toFixed(1) + ' KB';
    }
    return (size / (1024 * 1024)).toFixed(1) + ' MB';
  }

  function formatDate(value) {
    if (value === null || value === undefined || value === '') {
      return '';
    }
    var date = new Date(value);
    if (isNaN(date.getTime())) {
      return String(value);
    }
    return date.getFullYear() + '-' + pad2(date.getMonth() + 1) + '-' + pad2(date.getDate());
  }

  function pad2(value) {
    return value < 10 ? '0' + value : String(value);
  }

  function disposeCharts() {
    state.charts.forEach(function (chart) {
      try {
        chart.dispose();
      } catch (err) {
        /* 忽略销毁异常 */
      }
    });
    state.charts = [];
  }

  function resizeAllCharts() {
    if (D.section && D.section.hidden) {
      return;
    }
    state.charts.forEach(function (chart) {
      try {
        chart.resize();
      } catch (err) {
        /* 忽略尺寸异常 */
      }
    });
  }

  function galleryVisible() {
    return D.section ? !D.section.hidden : true;
  }

  function fetchJSON(url, options) {
    return fetch(url, options).then(function (response) {
      return response.text().then(function (text) {
        var data = null;
        if (text) {
          try {
            data = JSON.parse(text);
          } catch (err) {
            data = null;
          }
        }
        if (!response.ok) {
          var message = 'HTTP ' + response.status;
          if (data && (data.error || data.message)) {
            message = data.error || data.message;
          }
          throw new Error(message);
        }
        if (data === null) {
          throw new Error('服务器返回了无法解析的数据');
        }
        return data;
      });
    });
  }

  function toast(message, type, duration) {
    if (window.LeafApp && typeof window.LeafApp.toast === 'function') {
      window.LeafApp.toast(message, type, duration);
      return;
    }
    if (window.console && typeof window.console.log === 'function') {
      window.console.log(message);
    }
  }

  /* ---------- 窗口尺寸 ---------- */

  var resizePending = false;
  window.addEventListener('resize', function () {
    if (resizePending || !galleryVisible()) {
      return;
    }
    resizePending = true;
    window.requestAnimationFrame(function () {
      resizePending = false;
      resizeAllCharts();
    });
  });

  window.addEventListener('orientationchange', function () {
    window.requestAnimationFrame(resizeAllCharts);
  });

  /* ---------- 暴露调试接口 ---------- */

  window.LeafGallery = {
    reload: function () {
      ensureLoaded(true);
    },
    state: state
  };
})();
