/* ============================================================
   知叶 — pic 数据看板（plans.md §5.7 / §14.8 T8.7–T8.8）
   只渲染后端 /api/gallery 返回的 charts[]，使用本地 ECharts 5.x。
   ============================================================ */

(function () {
  'use strict';

  /* 固定配色：A 蓝、B 橙、C 绿、D 红、E 紫 */
  var MODEL_COLORS = {
    A: '#2f6fed',
    B: '#f28c28',
    C: '#2e9e5b',
    D: '#d64545',
    E: '#8e5cc7'
  };

  /* groupedBar 的系列是数据集（spec5 / live5 / …），使用协调色 */
  var DATASET_COLORS = ['#2f6fed', '#f28c28', '#2e9e5b', '#d64545', '#8e5cc7', '#7a6ff0'];

  var AXIS_COLOR = '#64776c';
  var TEXT_COLOR = '#3c4f44';
  var GRID_LINE = '#eef4ef';

  var D = {};

  var state = {
    loaded: false,
    loading: false,
    pendingRender: false,
    pendingTarget: 0,
    directories: [],
    activeIndex: -1,
    charts: []
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

      var directories = Array.isArray(data.directories) ? data.directories.slice() : [];
      directories.sort(function (a, b) {
        return compareNames(a && a.name, b && b.name);
      });
      state.directories = directories;
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
          if (!found && drawableCharts(dir).length) {
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
     目录 Tab（数量 = 图表数）
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
      var button = document.createElement('button');
      button.type = 'button';
      button.className = 'gallery-tab';
      button.id = 'gallery-tab-' + index;
      button.setAttribute('role', 'tab');
      button.setAttribute('aria-selected', index === state.activeIndex ? 'true' : 'false');
      button.setAttribute('aria-controls', D.panel && D.panel.id ? D.panel.id : 'galleryPanel');
      button.tabIndex = index === state.activeIndex ? 0 : -1;
      button.setAttribute('data-index', String(index));

      var label = document.createElement('span');
      label.textContent = dir.name || '(未命名目录)';
      var countValue = drawableCharts(dir).length;
      var count = document.createElement('span');
      count.className = 'gallery-tab-count';
      count.textContent = String(countValue);
      count.setAttribute('aria-label', countValue + ' 张图表');

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
     目录内容：只渲染图表卡片
     ============================================================ */

  function renderDirectory(dir) {
    if (!D.panel) {
      return;
    }
    D.panel.textContent = '';

    var charts = drawableCharts(dir);
    if (!charts.length) {
      showEmptyPanel('「' + (dir && dir.name ? dir.name : '') + '」暂无可绘制图表');
      return;
    }

    var grid = document.createElement('div');
    grid.className = 'chart-grid';

    var entries = charts.map(function (spec) {
      var card = document.createElement('section');
      card.className = 'chart-card';

      var title = document.createElement('h4');
      title.textContent = spec.title || '未命名图表';

      var element = document.createElement('div');
      element.className = 'chart';
      element.setAttribute('role', 'img');
      element.setAttribute('aria-label', spec.title || '图表');

      card.appendChild(title);
      card.appendChild(element);
      grid.appendChild(card);

      return { element: element, spec: spec };
    });

    D.panel.appendChild(grid);

    entries.forEach(function (entry) {
      initChart(entry.element, entry.spec);
    });
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

  function chartList(dir) {
    return dir && Array.isArray(dir.charts) ? dir.charts : [];
  }

  function drawableCharts(dir) {
    return chartList(dir).filter(isDrawableChart);
  }

  function isDrawableChart(spec) {
    return Boolean(
      spec &&
      typeof spec === 'object' &&
      Array.isArray(spec.x) &&
      spec.x.length &&
      Array.isArray(spec.series) &&
      spec.series.length
    );
  }

  /* ============================================================
     ECharts 通用渲染器
     ============================================================ */

  function initChart(element, spec) {
    if (!window.echarts) {
      element.className = 'chart chart-empty';
      element.textContent = 'ECharts 未加载';
      return;
    }
    try {
      var chart = window.echarts.init(element, null, { renderer: 'canvas' });
      chart.setOption(buildOption(spec));
      state.charts.push(chart);
    } catch (err) {
      element.className = 'chart chart-empty';
      element.textContent = '图表渲染失败：' + err.message;
    }
  }

  function buildOption(spec) {
    var type = spec.type === 'bar' || spec.type === 'groupedBar' ? spec.type : 'line';
    var unit = spec.unit === 'percent' ? 'percent' : 'value';
    var isLine = type === 'line';
    var isGrouped = type === 'groupedBar';

    var seriesNames = spec.series.map(function (item) {
      return String(item.name || item.key || '');
    });
    var colors = spec.series.map(function (item, index) {
      return seriesColor(item, index, isGrouped);
    });

    return {
      color: colors,
      textStyle: {
        fontFamily: '"Microsoft YaHei", "PingFang SC", sans-serif',
        color: TEXT_COLOR
      },
      animationDuration: 420,
      grid: {
        left: 56,
        right: 20,
        top: 56,
        bottom: 44,
        containLabel: true
      },
      legend: {
        top: 0,
        right: 8,
        type: 'scroll',
        itemWidth: 12,
        itemHeight: 8,
        itemGap: 12,
        textStyle: { color: TEXT_COLOR, fontSize: 10 },
        data: seriesNames
      },
      tooltip: buildTooltip(type, unit, spec),
      xAxis: {
        type: 'category',
        data: spec.x,
        boundaryGap: !isLine,
        axisTick: { alignWithLabel: true },
        axisLine: { lineStyle: { color: '#c3d8c9' } },
        axisLabel: {
          color: AXIS_COLOR,
          interval: 0,
          rotate: 0,
          fontSize: 10,
          formatter: categoryFormatter(isLine, spec.x.length)
        }
      },
      yAxis: buildYAxis(unit),
      series: spec.series.map(function (item, index) {
        return buildSeries(item, type, unit, spec, colors[index]);
      })
    };
  }

  function buildTooltip(type, unit, spec) {
    var tooltip = {
      trigger: 'axis',
      confine: true,
      axisPointer: { type: type === 'line' ? 'line' : 'shadow' }
    };

    /* 单指标柱状图：每个系列只有自身分类有值，过滤掉 null，避免悬停时满屏“—” */
    if (type === 'bar') {
      tooltip.formatter = function (params) {
        var rows = (Array.isArray(params) ? params : [params]).filter(function (item) {
          return item && item.value !== null && item.value !== undefined && item.value !== '';
        });
        var title = rows.length ? rows[0].axisValueLabel : '';
        var body = rows.map(function (item) {
          return '<div style="display:flex;align-items:center;gap:6px;">' +
            item.marker +
            '<span>' + escapeHtml(item.seriesName) + '</span>' +
            '<b style="margin-left:auto;">' +
            escapeHtml(formatTooltipValue(item.value, unit, spec)) +
            '</b></div>';
        }).join('');
        return '<div style="font-size:12px;line-height:1.7;min-width:140px;">' +
          '<div style="font-weight:600;margin-bottom:3px;">' + escapeHtml(title) + '</div>' +
          body + '</div>';
      };
    } else {
      tooltip.valueFormatter = function (value) {
        return formatTooltipValue(value, unit, spec);
      };
    }

    return tooltip;
  }

  function escapeHtml(value) {
    return String(value === null || value === undefined ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function buildYAxis(unit) {
    var axis = {
      type: 'value',
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: AXIS_COLOR, fontSize: 10 },
      splitLine: { lineStyle: { color: GRID_LINE } }
    };

    if (unit === 'percent') {
      axis.min = 0;
      axis.max = 1;
      axis.axisLabel.formatter = function (value) {
        return Math.round(value * 100) + '%';
      };
    }

    return axis;
  }

  function buildSeries(item, type, unit, spec, color) {
    var name = String(item.name || item.key || '');
    var data = Array.isArray(item.data) ? item.data : [];

    if (type === 'line') {
      return {
        name: name,
        type: 'line',
        data: data,
        smooth: false,
        showSymbol: false,
        symbol: 'none',
        connectNulls: false,
        lineStyle: { width: 2, color: color },
        itemStyle: { color: color },
        label: { show: false },
        emphasis: { focus: 'series' }
      };
    }

    var grouped = type === 'groupedBar';
    return {
      name: name,
      type: 'bar',
      data: data,
      barMaxWidth: grouped ? 26 : 44,
      barGap: grouped ? '10%' : '-100%',
      itemStyle: {
        color: color,
        borderRadius: grouped ? [4, 4, 0, 0] : [6, 6, 0, 0]
      },
      label: {
        show: spec.showDataLabels === true,
        position: 'top',
        color: TEXT_COLOR,
        fontSize: 10,
        formatter: function (params) {
          if (params.value === null || params.value === undefined) {
            return '';
          }
          var number = Number(params.value);
          if (!isFinite(number)) {
            return '';
          }
          return unit === 'percent'
            ? (number * 100).toFixed(1) + '%'
            : number.toFixed(3);
        }
      }
    };
  }

  function seriesColor(item, index, grouped) {
    if (grouped) {
      return DATASET_COLORS[index % DATASET_COLORS.length];
    }
    var key = String(item.key || '').toUpperCase();
    return MODEL_COLORS[key] || DATASET_COLORS[index % DATASET_COLORS.length];
  }

  function formatTooltipValue(value, unit, spec) {
    if (value === null || value === undefined || value === '') {
      return '—';
    }
    var number = Number(value);
    if (!isFinite(number)) {
      return '—';
    }
    if (unit === 'percent') {
      return (number * 100).toFixed(2) + '%';
    }
    if (spec && /gain/i.test(String(spec.id || ''))) {
      return number.toFixed(2) + ' 百分点';
    }
    return number.toFixed(3);
  }

  /*
   * 横坐标标签：柱状图始终显示全部分类；折线图轮次较多，
   * 用 formatter 只显示首末与每 5 轮的标签，避免文字重叠。
   * 只影响显示，interval 仍为 0，数据本身不变。
   */
  function categoryFormatter(isLine, count) {
    if (!isLine) {
      return formatCategoryLabel;
    }
    return function (value, index) {
      if (index === 0 || index % 5 === 0) {
        return String(value);
      }
      /* 末尾轮次与前一个标签太近时不再单独显示，避免重叠 */
      if (index === count - 1 && (count - 1) % 5 >= 2) {
        return String(value);
      }
      return '';
    };
  }

  function formatCategoryLabel(value) {
    var text = String(value === null || value === undefined ? '' : value);
    /* 只对超长标签换行，不修改数据本身 */
    if (text.length > 12 && text.indexOf(' (') > 0) {
      return text.replace(' (', '\n(');
    }
    return text;
  }

  /* ============================================================
     工具
     ============================================================ */

  function compareNames(a, b) {
    return String(a).localeCompare(String(b), 'zh-CN', {
      numeric: true,
      sensitivity: 'base'
    });
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
