/* ============================================================
   知叶 — 识别页逻辑
   原生 JavaScript ES6，全局命名空间 window.LeafApp
   ============================================================ */

(function () {
  'use strict';

  /* ---------- 静态类别回退表（来自 data/class_registry*.json） ---------- */

  var FIVE_CLASSES = [
    { key: 'cercis_chinensis', zh: '紫荆', sci: 'Cercis chinensis' },
    { key: 'pittosporum_tobira', zh: '海桐', sci: 'Pittosporum tobira (Thunb.) Ait. f.' },
    { key: 'cinnamomum_camphora', zh: '香樟', sci: 'Cinnamomum camphora (L.) J. Presl' },
    { key: 'osmanthus_fragrans', zh: '桂花', sci: 'Osmanthus fragrans Lour.' },
    { key: 'lagerstroemia_indica', zh: '紫薇', sci: 'Lagerstroemia indica (L.) Pers.' }
  ];

  var ALL32_CLASSES = [
    { key: 'phyllostachys_edulis', zh: '毛竹', sci: 'Phyllostachys edulis (Carr.) Houz.' },
    { key: 'aesculus_chinensis', zh: '七叶树', sci: 'Aesculus chinensis' },
    { key: 'berberis_anhweiensis', zh: '安徽小檗', sci: 'Berberis anhweiensis Ahrendt' },
    { key: 'cercis_chinensis', zh: '紫荆', sci: 'Cercis chinensis' },
    { key: 'indigofera_tinctoria', zh: '木蓝', sci: 'Indigofera tinctoria L.' },
    { key: 'acer_palmatum', zh: '鸡爪槭', sci: 'Acer Palmatum' },
    { key: 'phoebe_nanmu', zh: '楠木', sci: 'Phoebe nanmu (Oliv.) Gamble' },
    { key: 'kalopanax_septemlobus', zh: '刺楸', sci: 'Kalopanax septemlobus (Thunb. ex A.Murr.) Koidz.' },
    { key: 'cinnamomum_japonicum', zh: '天竺桂', sci: 'Cinnamomum japonicum Sieb.' },
    { key: 'koelreuteria_paniculata', zh: '栾树', sci: 'Koelreuteria paniculata Laxm.' },
    { key: 'ilex_macrocarpa', zh: '大果冬青', sci: 'Ilex macrocarpa Oliv.' },
    { key: 'pittosporum_tobira', zh: '海桐', sci: 'Pittosporum tobira (Thunb.) Ait. f.' },
    { key: 'chimonanthus_praecox', zh: '蜡梅', sci: 'Chimonanthus praecox L.' },
    { key: 'cinnamomum_camphora', zh: '香樟', sci: 'Cinnamomum camphora (L.) J. Presl' },
    { key: 'viburnum_awabuki', zh: '日本珊瑚树', sci: 'Viburnum awabuki K.Koch' },
    { key: 'osmanthus_fragrans', zh: '桂花', sci: 'Osmanthus fragrans Lour.' },
    { key: 'cedrus_deodara', zh: '雪松', sci: 'Cedrus deodara (Roxb.) G. Don' },
    { key: 'ginkgo_biloba', zh: '银杏', sci: 'Ginkgo biloba L.' },
    { key: 'lagerstroemia_indica', zh: '紫薇', sci: 'Lagerstroemia indica (L.) Pers.' },
    { key: 'nerium_oleander', zh: '夹竹桃', sci: 'Nerium oleander L.' },
    { key: 'podocarpus_macrophyllus', zh: '罗汉松', sci: 'Podocarpus macrophyllus (Thunb.) Sweet' },
    { key: 'prunus_serrulata', zh: '日本晚樱', sci: 'Prunus serrulata Lindl. var. lannesiana auct.' },
    { key: 'ligustrum_lucidum', zh: '女贞', sci: 'Ligustrum lucidum Ait. f.' },
    { key: 'toona_sinensis', zh: '香椿', sci: 'Tonna sinensis M. Roem.' },
    { key: 'prunus_persica', zh: '桃', sci: 'Prunus persica (L.) Batsch' },
    { key: 'manglietia_fordiana', zh: '木莲', sci: 'Manglietia fordiana Oliv.' },
    { key: 'acer_buergerianum', zh: '三角枫', sci: 'Acer buergerianum Miq.' },
    { key: 'mahonia_bealei', zh: '阔叶十大功劳', sci: 'Mahonia bealei (Fortune) Carr.' },
    { key: 'magnolia_grandiflora', zh: '广玉兰', sci: 'Magnolia grandiflora L.' },
    { key: 'populus_canadensis', zh: '加杨', sci: 'Populus ×canadensis Moench' },
    { key: 'liriodendron_chinense', zh: '鹅掌楸', sci: 'Liriodendron chinense (Hemsl.) Sarg.' },
    { key: 'citrus_reticulata', zh: '柑橘', sci: 'Citrus reticulata Blanco' }
  ];

  var MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
  var CAMERA_INTERVAL_MS = 700;

  var D = {};

  var state = {
    groups: [],
    models: [],
    group: null,
    model: null,
    modelsReady: false,
    activeSection: 'recognize',
    uploadSeq: 0,
    cameraSeq: 0,
    uploadFile: null,
    uploadPreviewUrl: null,
    uploading: false,
    demoBusy: false,
    lastResult: null,
    resultPending: false,
    gradcamPending: false,
    camera: {
      stream: null,
      timer: null,
      running: false,
      starting: false,
      startToken: 0,
      paused: false,
      inFlight: false,
      photoBusy: false,
      photoSeq: 0,
      errors: 0
    },
    modal: {
      resolve: null,
      mode: 'confirm',
      trigger: null
    }
  };

  /* ============================================================
     入口
     ============================================================ */

  document.addEventListener('DOMContentLoaded', init);

  function init() {
    cacheDom();
    bindNavigation();
    bindTabs();
    bindCamera();
    bindUpload();
    bindGradcam();
    bindDemo();
    bindModal();
    bindLegendSearch();

    renderGroupSwitch();
    renderModels();
    renderLegend();
    updateCameraControls();
    activateSection('recognize', true);
    loadModels();
  }

  function cacheDom() {
    D.navToggle = document.getElementById('navToggle');
    D.primaryNav = document.getElementById('primaryNav');
    D.groupSwitch = document.getElementById('groupSwitch');
    D.modelPills = document.getElementById('modelPills');
    D.gradcamToggle = document.getElementById('gradcamToggle');
    D.engineBadge = document.getElementById('engineBadge');

    D.tabCamera = document.getElementById('tab-camera');
    D.tabUpload = document.getElementById('tab-upload');
    D.panelCamera = document.getElementById('panel-camera');
    D.panelUpload = document.getElementById('panel-upload');

    D.cameraStage = document.getElementById('cameraStage');
    D.cameraVideo = document.getElementById('cameraVideo');
    D.cameraSnapshot = document.getElementById('cameraSnapshot');
    D.cameraPlaceholder = document.getElementById('cameraPlaceholder');
    D.cameraStatus = document.getElementById('cameraStatus');
    D.cameraStartBtn = document.getElementById('cameraStartBtn');
    D.cameraStopBtn = document.getElementById('cameraStopBtn');
    D.cameraPhotoBtn = document.getElementById('cameraPhotoBtn');
    D.captureCanvas = document.getElementById('captureCanvas');

    D.dropzone = document.getElementById('dropzone');
    D.fileInput = document.getElementById('fileInput');
    D.uploadPreviewWrap = document.getElementById('uploadPreviewWrap');
    D.uploadPreview = document.getElementById('uploadPreview');
    D.uploadFileName = document.getElementById('uploadFileName');
    D.clearUploadBtn = document.getElementById('clearUploadBtn');
    D.uploadStatus = document.getElementById('uploadStatus');

    D.resultEmpty = document.getElementById('resultEmpty');
    D.resultBody = document.getElementById('resultBody');
    D.resultZh = document.getElementById('resultZh');
    D.resultSci = document.getElementById('resultSci');
    D.resultModelTag = document.getElementById('resultModelTag');
    D.resultLatency = document.getElementById('resultLatency');
    D.lowConfidenceWarning = document.getElementById('lowConfidenceWarning');
    D.topkList = document.getElementById('topkList');
    D.gradcamPanel = document.getElementById('gradcamPanel');
    D.gradcamImage = document.getElementById('gradcamImage');
    D.gradcamUnavailable = document.getElementById('gradcamUnavailable');

    D.legendSearch = document.getElementById('legendSearch');
    D.legendCount = document.getElementById('legendCount');
    D.classLegend = document.getElementById('classLegend');

    D.addDataBtn = document.getElementById('addDataBtn');
    D.trainBtn = document.getElementById('trainBtn');

    D.toastContainer = document.getElementById('toastContainer');
    D.modalBackdrop = document.getElementById('modalBackdrop');
    D.modalTitle = document.getElementById('modalTitle');
    D.modalText = document.getElementById('modalText');
    D.modalActions = document.getElementById('modalActions');
    D.modalCancel = document.getElementById('modalCancel');
    D.modalConfirm = document.getElementById('modalConfirm');
    D.trainProgressWrap = document.getElementById('trainProgressWrap');
    D.trainProgressBar = document.getElementById('trainProgressBar');
    D.trainProgressText = document.getElementById('trainProgressText');
  }

  /* ============================================================
     导航与 Section 切换
     ============================================================ */

  function bindNavigation() {
    var links = document.querySelectorAll('[data-section-link]');
    Array.prototype.forEach.call(links, function (link) {
      link.addEventListener('click', function (event) {
        event.preventDefault();
        activateSection(link.getAttribute('data-section-link'));
      });
    });

    if (D.navToggle) {
      D.navToggle.addEventListener('click', function () {
        setNavOpen(D.navToggle.getAttribute('aria-expanded') !== 'true');
      });
    }

    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape') {
        setNavOpen(false);
      }
    });

    document.addEventListener('click', function (event) {
      if (!D.primaryNav || !D.primaryNav.classList.contains('is-open')) {
        return;
      }
      var header = D.primaryNav.parentNode;
      if (header && !header.contains(event.target)) {
        setNavOpen(false);
      }
    });

    window.addEventListener('resize', function () {
      if (window.innerWidth > 760) {
        setNavOpen(false);
      }
    });
  }

  function setNavOpen(open) {
    if (!D.navToggle || !D.primaryNav) {
      return;
    }
    D.navToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    D.primaryNav.classList.toggle('is-open', open);
    var label = D.navToggle.querySelector('.sr-only');
    if (label) {
      label.textContent = open ? '关闭导航菜单' : '打开导航菜单';
    }
  }

  function activateSection(name, silent) {
    var map = {
      recognize: 'section-recognize',
      gallery: 'section-gallery',
      system: 'section-system'
    };
    if (!map[name]) {
      return;
    }

    Object.keys(map).forEach(function (key) {
      var section = document.getElementById(map[key]);
      if (section) {
        section.hidden = key !== name;
      }
    });

    var links = document.querySelectorAll('[data-section-link]');
    Array.prototype.forEach.call(links, function (link) {
      var isActive = link.getAttribute('data-section-link') === name;
      link.classList.toggle('is-active', isActive);
      if (isActive) {
        link.setAttribute('aria-current', 'page');
      } else {
        link.removeAttribute('aria-current');
      }
    });

    state.activeSection = name;
    if (name !== 'recognize') {
      stopCamera();
    }
    setNavOpen(false);

    if (!silent) {
      try {
        window.scrollTo({ top: 0, behavior: 'smooth' });
      } catch (err) {
        window.scrollTo(0, 0);
      }
    }

    document.dispatchEvent(new CustomEvent('app:sectionchange', {
      detail: { section: name }
    }));
  }

  /* ============================================================
     Tab（摄像头 / 上传）
     ============================================================ */

  function bindTabs() {
    var tablist = document.querySelector('.tab-list[role="tablist"]');
    if (!tablist) {
      return;
    }
    var tabs = tablist.querySelectorAll('[role="tab"]');
    Array.prototype.forEach.call(tabs, function (tab, index) {
      tab.addEventListener('click', function () {
        activateTab(tab);
      });
      tab.addEventListener('keydown', function (event) {
        handleTabKeys(event, tabs, index, activateTab);
      });
    });
  }

  function activateTab(tab) {
    var tablist = tab.parentNode;
    var tabs = tablist.querySelectorAll('[role="tab"]');

    Array.prototype.forEach.call(tabs, function (item) {
      var selected = item === tab;
      item.setAttribute('aria-selected', selected ? 'true' : 'false');
      item.classList.toggle('is-active', selected);
      item.tabIndex = selected ? 0 : -1;

      var panel = document.getElementById(item.getAttribute('aria-controls'));
      if (panel) {
        panel.hidden = !selected;
      }
    });

    if (tab.getAttribute('data-tab') === 'upload') {
      if (state.camera.running || state.camera.starting || state.camera.stream) {
        stopCamera();
      } else {
        state.camera.startToken += 1;
      }
    }
  }

  function handleTabKeys(event, tabs, index, activate) {
    var last = tabs.length - 1;
    var next = -1;

    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
      next = index === last ? 0 : index + 1;
    } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
      next = index === 0 ? last : index - 1;
    } else if (event.key === 'Home') {
      next = 0;
    } else if (event.key === 'End') {
      next = last;
    }

    if (next >= 0) {
      event.preventDefault();
      tabs[next].focus();
      activate(tabs[next]);
    }
  }

  /* ============================================================
     模型加载 / 模型组 / 模型 pill
     ============================================================ */

  function loadModels() {
    setEngineBadge('正在加载模型…', false);

    return fetchJSON('/api/models').then(function (data) {
      state.groups = Array.isArray(data.groups) ? data.groups : [];
      state.models = Array.isArray(data.models) ? data.models : [];

      if (!state.groups.length) {
        throw new Error('后端没有返回任何模型组');
      }

      state.modelsReady = true;
      renderGroupSwitch();

      var preferred = null;
      state.groups.forEach(function (group) {
        if (!preferred && Number(group.num_classes) <= 5) {
          preferred = group;
        }
      });
      if (!preferred) {
        preferred = state.groups[0];
      }

      selectGroup(preferred.id, true);

      var available = state.models.filter(function (model) {
        return model.available;
      });
      if (available.length) {
        setEngineBadge('已加载 ' + available.length + ' 个可用模型', false);
      } else {
        setEngineBadge('暂无可用模型', true);
        toast('未发现可用模型，请确认 runs/ 下已存在 best.pt', 'warn', 5600);
      }
    }).catch(function (err) {
      state.modelsReady = false;
      setEngineBadge('模型加载失败', true);
      renderGroupSwitch();
      renderModels();
      renderLegend();
      toast('模型加载失败：' + err.message, 'error', 6500);
    });
  }

  function setEngineBadge(text, isError) {
    if (!D.engineBadge) {
      return;
    }
    D.engineBadge.textContent = text;
    D.engineBadge.classList.toggle('is-error', Boolean(isError));
  }

  function renderGroupSwitch() {
    if (!D.groupSwitch) {
      return;
    }
    D.groupSwitch.textContent = '';

    state.groups.forEach(function (group) {
      var button = document.createElement('button');
      button.type = 'button';
      button.className = 'group-tab';
      button.setAttribute('role', 'tab');
      button.setAttribute('aria-selected', state.group && state.group.id === group.id ? 'true' : 'false');
      button.setAttribute('data-group', group.id);
      button.textContent = group.label || group.id;
      if (group.num_classes) {
        button.title = '共 ' + group.num_classes + ' 类';
      }
      button.addEventListener('click', function () {
        selectGroup(group.id);
      });
      D.groupSwitch.appendChild(button);
    });

    var tabs = D.groupSwitch.querySelectorAll('.group-tab');
    Array.prototype.forEach.call(tabs, function (tab, index) {
      tab.addEventListener('keydown', function (event) {
        var last = tabs.length - 1;
        var next = -1;
        if (event.key === 'ArrowRight') {
          next = index === last ? 0 : index + 1;
        } else if (event.key === 'ArrowLeft') {
          next = index === 0 ? last : index - 1;
        }
        if (next >= 0) {
          event.preventDefault();
          tabs[next].focus();
          tabs[next].click();
        }
      });
    });
  }

  function selectGroup(groupId, silent) {
    var group = null;
    state.groups.forEach(function (item) {
      if (item.id === groupId) {
        group = item;
      }
    });
    if (!group) {
      return;
    }

    var changed = !state.group || state.group.id !== group.id;
    state.group = group;

    var models = modelsInGroup();
    var previousId = state.model ? state.model.id : '';
    var nextModel = null;
    /* 新组里仍有同名可用模型时保持当前选择 */
    models.forEach(function (model) {
      if (!nextModel && model.available && model.id === previousId) {
        nextModel = model;
      }
    });
    /* 否则退回到新组第一个可用模型 */
    models.forEach(function (model) {
      if (!nextModel && model.available) {
        nextModel = model;
      }
    });
    state.model = nextModel;

    renderGroupSwitch();
    renderModels();
    renderLegend();

    /* 点击已选中的组：只刷新 UI，不作废在途的拍照/上传重识别请求 */
    if (!changed) {
      return;
    }

    /* 丢弃切换前在途的摄像头帧结果 */
    state.cameraSeq += 1;

    if (silent) {
      return;
    }

    if (!models.length || !state.model) {
      resetResult('该模型组暂无可用模型');
      /* 作废在途上传请求，避免迟到响应把旧组结果显示到无模型的新组上；
         同时手动解除 busy（finalizer 因 seq 不匹配不会再执行） */
      state.uploadSeq += 1;
      state.uploading = false;
      setUploadBusy(false);
      toast(models.length ? '该模型组暂无可用模型' : '该模型组暂无模型', 'warn', 4000);
      /* 实时/拍照中切到无可用模型的组：明确提示识别已暂停 */
      if (state.camera.running) {
        setCameraStatus(state.modelsReady
          ? '当前模型组暂无可用模型，识别已暂停'
          : '模型尚未加载完成，识别已暂停',
          state.modelsReady ? 'error' : '');
      }
      return;
    }
    /* 保留当前上传图片/拍照快照，用新组选中的模型重新识别 */
    if (!reRecognizeCurrent()) {
      resetResult('请先选择图片或拍照，切换模型后将自动重新识别');
    }
  }

  function modelsInGroup() {
    if (!state.group) {
      return [];
    }
    var matched = state.models.filter(function (model) {
      return model.group === state.group.id || model.group_label === state.group.label;
    });
    if (matched.length) {
      return matched;
    }
    var targetCount = Number(state.group.num_classes);
    return state.models.filter(function (model) {
      return !model.group && Number(model.num_classes) === targetCount;
    });
  }

  function renderModels() {
    if (!D.modelPills) {
      return;
    }
    D.modelPills.textContent = '';

    var models = modelsInGroup();
    models.forEach(function (model) {
      var button = document.createElement('button');
      button.type = 'button';
      button.className = 'model-pill';
      button.setAttribute('aria-pressed',
        state.model && state.model.id === model.id && state.model.group === model.group
          ? 'true'
          : 'false');
      button.disabled = !model.available;

      var name = document.createElement('span');
      name.className = 'pill-name';
      name.textContent = model.name || model.id;

      var metric = document.createElement('span');
      metric.className = 'pill-metric';
      var accuracy = (model.accuracy === null || model.accuracy === undefined)
        ? '—'
        : formatPct(model.accuracy);
      metric.textContent = '准确率 ' + accuracy + (model.available ? '' : ' · 不可用');

      button.appendChild(name);
      button.appendChild(metric);

      if (model.available) {
        button.title = (model.name || model.id) +
          ' · accuracy ' + formatPct(model.accuracy) +
          ' · macro-F1 ' + formatPct(model.macro_f1);
      } else {
        button.title = model.error || '缺少模型权重，暂不可用';
      }

      button.addEventListener('click', function () {
        selectModel(model.id);
      });

      D.modelPills.appendChild(button);
    });
  }

  function selectModel(modelId) {
    var model = null;
    if (!state.group) {
      return;
    }
    modelsInGroup().forEach(function (item) {
      if (!model && item.id === modelId && item.group === state.group.id) {
        model = item;
      }
    });
    if (!model) {
      return;
    }
    if (!model.available) {
      toast(model.error || ((model.name || model.id) + ' 暂不可用'), 'warn', 4500);
      return;
    }
    if (state.model &&
        state.model.id === model.id &&
        state.model.group === model.group) {
      return;
    }

    state.model = model;
    state.cameraSeq += 1;
    renderModels();

    /* 不清除当前上传预览/拍照快照，直接用新模型重新识别 */
    if (!reRecognizeCurrent()) {
      resetResult('请先选择图片或拍照，切换模型后将自动重新识别');
    }
  }

  /* 切换模型/模型组后的统一重识别入口。
     有拍照快照或上传图片时自动用当前模型重新请求；
     实时摄像头交给下一帧自然使用新模型；没有素材时只提示，不伪造结果。 */
  function reRecognizeCurrent(reason) {
    var gradcamOnly = reason === 'gradcam';
    if (!state.modelsReady || !state.group || !state.model) {
      return false;
    }

    if (cameraPanelActive()) {
      var snapshot = currentCameraSnapshot();
      if (snapshot) {
        if (!gradcamOnly) {
          markResultPending();
        } else {
          showGradcamPending();
        }
        recognizeSnapshot(snapshot, true, reason);
        return true;
      }
      if (state.camera.running || state.camera.starting) {
        /* 实时轮询会自然带上新模型/新开关状态，不额外重发，避免与在途帧重叠 */
        setCameraStatus(gradcamOnly
          ? 'Grad-CAM 已开启，下一帧将显示热力图'
          : '已切换模型，下一帧将使用新模型', '');
        if (gradcamOnly) {
          showGradcamPending();
          /* 已在途的帧返回后会立刻安排下一帧，这里只加速尚未开始的轮询 */
          scheduleCameraTick(0);
        }
        return true;
      }
      /* 摄像头页未开启且无快照：不回退去识别看不见的上传图片 */
      return false;
    }

    if (state.uploadFile) {
      if (!gradcamOnly) {
        markResultPending();
      } else {
        showGradcamPending();
      }
      predictFile(state.uploadFile, true, reason);
      return true;
    }
    return false;
  }

  function topkForGroup() {
    var count = state.group ? Number(state.group.num_classes) : 5;
    return count > 10 ? 5 : 3;
  }

  /* ============================================================
     类别图例
     ============================================================ */

  function classesForGroup() {
    var group = state.group;
    if (group && Array.isArray(group.classes) && group.classes.length) {
      return group.classes.map(function (item) {
        return {
          key: item.key || '',
          zh: item.zh_name || item.zh || item.name || '',
          sci: item.scientific_name || item.scientific || item.sci || ''
        };
      });
    }
    var count = group ? Number(group.num_classes) : 5;
    return count > 10 ? ALL32_CLASSES : FIVE_CLASSES;
  }

  function renderLegend() {
    if (!D.classLegend) {
      return;
    }
    D.classLegend.textContent = '';
    var classes = classesForGroup();

    if (D.legendCount) {
      D.legendCount.textContent = classes.length ? classes.length + ' 类' : '';
    }
    if (D.legendSearch) {
      D.legendSearch.hidden = classes.length <= 10;
      D.legendSearch.value = '';
    }
    if (D.classLegend) {
      D.classLegend.classList.toggle('is-scrollable', classes.length > 10);
    }

    if (!classes.length) {
      var empty = document.createElement('li');
      empty.className = 'legend-empty';
      empty.textContent = '暂无类别信息';
      D.classLegend.appendChild(empty);
      return;
    }

    classes.forEach(function (item) {
      var li = document.createElement('li');
      li.className = 'legend-item';
      li.setAttribute('data-search', (item.key + ' ' + item.zh + ' ' + item.sci).toLowerCase());
      li.title = (item.zh || item.key || '未知') + (item.sci ? ' — ' + item.sci : '');

      var zh = document.createElement('strong');
      zh.textContent = item.zh || item.key || '未知';
      var sci = document.createElement('span');
      sci.textContent = item.sci || item.key || '';

      li.appendChild(zh);
      li.appendChild(sci);
      D.classLegend.appendChild(li);
    });
  }

  function bindLegendSearch() {
    if (!D.legendSearch) {
      return;
    }
    D.legendSearch.addEventListener('input', function () {
      filterLegend(this.value);
    });
  }

  function filterLegend(query) {
    if (!D.classLegend) {
      return;
    }
    var needle = String(query || '').trim().toLowerCase();
    var items = D.classLegend.querySelectorAll('.legend-item');
    var visible = 0;

    Array.prototype.forEach.call(items, function (item) {
      var haystack = item.getAttribute('data-search') || '';
      var match = !needle || haystack.indexOf(needle) !== -1;
      item.hidden = !match;
      if (match) {
        visible += 1;
      }
    });

    if (items.length && visible === 0) {
      var hint = D.classLegend.querySelector('.legend-empty');
      if (!hint) {
        hint = document.createElement('li');
        hint.className = 'legend-empty';
        D.classLegend.appendChild(hint);
      }
      hint.textContent = '没有匹配的类别';
      hint.hidden = false;
    } else {
      var existing = D.classLegend.querySelector('.legend-empty');
      if (existing) {
        existing.hidden = true;
      }
    }
  }

  /* ============================================================
     摄像头
     ============================================================ */

  function bindCamera() {
    if (D.cameraStartBtn) {
      D.cameraStartBtn.addEventListener('click', function () {
        if (state.camera.running && state.camera.paused) {
          resumeCamera();
        } else {
          startCamera();
        }
      });
    }
    if (D.cameraStopBtn) {
      D.cameraStopBtn.addEventListener('click', stopCamera);
    }
    if (D.cameraPhotoBtn) {
      D.cameraPhotoBtn.addEventListener('click', takePhoto);
    }

    document.addEventListener('visibilitychange', function () {
      if (!state.camera.running) {
        return;
      }
      if (document.hidden) {
        setCameraStatus('页面已切换到后台，实时识别暂停', '');
      } else if (!state.camera.paused) {
        setCameraStatus('实时识别中…', 'live');
      }
    });
  }

  function cameraPanelActive() {
    return Boolean(D.panelCamera) && !D.panelCamera.hidden;
  }

  function startCamera() {
    if (state.camera.running) {
      if (state.camera.paused) {
        resumeCamera();
      }
      return;
    }
    if (state.camera.starting) {
      return;
    }
    if (!cameraPanelActive()) {
      toast('请先切换到「摄像头」标签', 'warn', 3600);
      return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setCameraStatus('当前浏览器不支持摄像头，请改用上传图片', 'error');
      toast('当前浏览器不支持摄像头，请改用上传图片', 'error', 5200);
      return;
    }

    var token = ++state.camera.startToken;
    state.camera.starting = true;
    setCameraStatus('正在请求摄像头权限…', '');
    updateCameraControls();

    navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'environment' },
      audio: false
    }).then(function (stream) {
      if (token !== state.camera.startToken ||
          state.activeSection !== 'recognize' ||
          !cameraPanelActive()) {
        stopStreamTracks(stream);
        return false;
      }

      state.camera.stream = stream;
      D.cameraVideo.srcObject = stream;
      var playPromise = D.cameraVideo.play();
      if (playPromise && typeof playPromise.catch === 'function') {
        playPromise.catch(function () {
          /* 自动播放被阻止时仍可继续，由点按触发 */
        });
      }
      return true;
    }).then(function (accepted) {
      if (!accepted) {
        return;
      }
      if (token !== state.camera.startToken ||
          state.activeSection !== 'recognize' ||
          !cameraPanelActive()) {
        return;
      }

      state.camera.starting = false;
      state.camera.running = true;
      state.camera.paused = false;
      state.camera.inFlight = false;
      state.camera.errors = 0;
      D.cameraStage.classList.add('is-live');
      D.cameraPlaceholder.hidden = true;
      D.cameraSnapshot.hidden = true;
      D.cameraSnapshot.removeAttribute('src');
      setCameraStatus(state.modelsReady && state.model
        ? '实时识别中…'
        : '摄像头已开启，等待模型加载…', state.modelsReady && state.model ? 'live' : '');
      updateCameraControls();
      scheduleCameraTick(0);
    }).catch(function (err) {
      /* 启动已被 Stop / 切 Tab / 切分区作废：不处理过期错误 */
      if (token !== state.camera.startToken) {
        return;
      }

      state.camera.starting = false;
      var reason = describeCameraError(err);
      setCameraStatus('无法访问摄像头：' + reason, 'error');
      toast('无法访问摄像头：' + reason + '。上传图片仍可正常使用。', 'error', 7000);
      releaseCameraStream();
      state.camera.running = false;
      state.camera.paused = false;
      D.cameraStage.classList.remove('is-live');
      D.cameraPlaceholder.hidden = false;
      updateCameraControls();
    });
  }

  function describeCameraError(err) {
    var name = err && err.name ? err.name : '';
    if (name === 'NotAllowedError' || name === 'PermissionDeniedError' || name === 'SecurityError') {
      return '权限被拒绝。请允许浏览器访问摄像头，并通过 http://127.0.0.1:5000 或 http://localhost:5000 打开页面';
    }
    if (name === 'NotFoundError' || name === 'DevicesNotFoundError' || name === 'OverconstrainedError') {
      return '未检测到可用的摄像头设备';
    }
    if (name === 'NotReadableError' || name === 'TrackStartError') {
      return '摄像头可能被其他程序占用，请关闭占用程序后重试';
    }
    return (err && err.message) ? err.message : '未知错误';
  }

  function scheduleCameraTick(delay) {
    if (!state.camera.running) {
      return;
    }
    clearTimeout(state.camera.timer);
    var wait = typeof delay === 'number' ? delay : CAMERA_INTERVAL_MS;
    state.camera.timer = setTimeout(cameraTick, wait);
  }

  function cameraTick() {
    if (!state.camera.running) {
      return;
    }
    if (state.activeSection !== 'recognize' || !cameraPanelActive()) {
      scheduleCameraTick(500);
      return;
    }
    if (document.hidden) {
      scheduleCameraTick(500);
      return;
    }
    if (state.camera.paused) {
      scheduleCameraTick(400);
      return;
    }
    if (!state.modelsReady || !state.model) {
      setCameraStatus(state.modelsReady
        ? '当前模型组暂无可用模型，识别已暂停'
        : '模型尚未加载完成，识别已暂停',
        state.modelsReady ? 'error' : '');
      scheduleCameraTick(CAMERA_INTERVAL_MS);
      return;
    }
    if (state.camera.inFlight) {
      scheduleCameraTick(CAMERA_INTERVAL_MS);
      return;
    }

    var dataUrl;
    try {
      dataUrl = captureSquareDataUrl();
    } catch (err) {
      scheduleCameraTick(CAMERA_INTERVAL_MS);
      return;
    }

    state.camera.inFlight = true;
    sendFrame(dataUrl).then(function (result) {
      if (!result || !result.fresh) {
        return;
      }
      state.camera.errors = 0;
      if (!state.camera.paused && !document.hidden) {
        setCameraStatus('实时识别中…', 'live');
      }
    }).catch(function (err) {
      if (err && err.__cameraStale) {
        return;
      }
      state.camera.errors += 1;
      if (state.camera.errors === 1) {
        setCameraStatus('实时识别失败：' + err.message, 'error');
        toast('实时识别失败：' + err.message, 'error', 4500);
      }
      if (state.camera.errors >= 3) {
        stopCamera();
        toast('实时识别连续失败，已停止摄像头', 'error', 5200);
        return;
      }
    }).then(function () {
      state.camera.inFlight = false;
      scheduleCameraTick(CAMERA_INTERVAL_MS);
    });
  }

  function captureSquareDataUrl() {
    var video = D.cameraVideo;
    if (!video || video.readyState < 2 || !video.videoWidth || !video.videoHeight) {
      throw new Error('摄像头画面尚未就绪');
    }

    var width = video.videoWidth;
    var height = video.videoHeight;
    var side = Math.min(width, height);
    var canvas = D.captureCanvas;
    canvas.width = side;
    canvas.height = side;
    var context = canvas.getContext('2d');
    if (!context) {
      throw new Error('无法创建画布上下文');
    }
    var sx = (width - side) / 2;
    var sy = (height - side) / 2;
    context.drawImage(video, sx, sy, side, side, 0, 0, side, side);

    return canvas.toDataURL('image/jpeg', 0.9);
  }

  function resumeCamera() {
    if (!state.camera.running) {
      return;
    }
    state.camera.paused = false;
    state.camera.errors = 0;
    D.cameraSnapshot.hidden = true;
    D.cameraSnapshot.removeAttribute('src');
    if (state.modelsReady && state.model) {
      setCameraStatus('实时识别中…', 'live');
    } else {
      setCameraStatus(state.modelsReady
        ? '当前模型组暂无可用模型，识别已暂停'
        : '模型尚未加载完成，识别已暂停',
        state.modelsReady ? 'error' : '');
    }
    updateCameraControls();
    scheduleCameraTick(0);
  }

  function stopCamera() {
    /* 作废正在进行的启动请求与在途摄像头识别请求 */
    state.camera.startToken += 1;
    state.camera.starting = false;
    state.cameraSeq += 1;

    state.camera.running = false;
    state.camera.paused = false;
    state.camera.inFlight = false;
    state.camera.photoBusy = false;
    state.camera.errors = 0;

    clearTimeout(state.camera.timer);
    state.camera.timer = null;

    releaseCameraStream();
    if (D.cameraVideo) {
      D.cameraVideo.srcObject = null;
    }
    if (D.cameraStage) {
      D.cameraStage.classList.remove('is-live');
    }
    if (D.cameraPlaceholder) {
      D.cameraPlaceholder.hidden = false;
    }
    if (D.cameraSnapshot) {
      D.cameraSnapshot.hidden = true;
      D.cameraSnapshot.removeAttribute('src');
    }
    setCameraStatus('摄像头已关闭', '');
    settleGradcamAfterCameraStop();
    updateCameraControls();
  }

  function releaseCameraStream() {
    if (!state.camera.stream) {
      return;
    }
    stopStreamTracks(state.camera.stream);
    state.camera.stream = null;
  }

  function stopStreamTracks(stream) {
    if (!stream) {
      return;
    }
    try {
      stream.getTracks().forEach(function (track) {
        track.stop();
      });
    } catch (err) {
      /* 忽略停止失败 */
    }
  }

  function updateCameraControls() {
    if (!D.cameraStartBtn) {
      return;
    }
    var running = state.camera.running;
    var starting = state.camera.starting;
    var paused = state.camera.paused;

    D.cameraStartBtn.disabled = starting || (running && !paused);
    D.cameraStartBtn.textContent = starting
      ? '请求中…'
      : ((running && paused) ? '继续' : '开始');
    D.cameraStopBtn.disabled = !running && !starting;
    D.cameraPhotoBtn.disabled = !running || starting || paused || state.camera.photoBusy;
  }

  function setCameraStatus(text, kind) {
    if (!D.cameraStatus) {
      return;
    }
    var nextClass = 'camera-status';
    if (kind === 'error') {
      nextClass += ' is-error';
    } else if (kind === 'live') {
      nextClass += ' is-live';
    }
    if (D.cameraStatus.textContent === text && D.cameraStatus.className === nextClass) {
      return;
    }
    D.cameraStatus.textContent = text;
    D.cameraStatus.className = nextClass;
  }

  function takePhoto() {
    if (!state.camera.running) {
      toast('请先打开摄像头', 'warn');
      return;
    }
    if (state.camera.photoBusy) {
      return;
    }

    var dataUrl;
    try {
      dataUrl = captureSquareDataUrl();
    } catch (err) {
      toast('拍照失败：' + err.message, 'error', 4200);
      return;
    }

    state.camera.paused = true;
    D.cameraSnapshot.src = dataUrl;
    D.cameraSnapshot.hidden = false;
    D.cameraStage.classList.add('is-live');
    setCameraStatus('已拍照并暂停实时识别，点击「继续」恢复', '');
    updateCameraControls();

    if (!state.modelsReady || !state.model) {
      toast('模型尚未就绪，照片暂未识别', 'warn', 4200);
      resetResult('模型尚未就绪，照片未识别，请稍后重新拍照');
      return;
    }

    /* 新快照开始即进入 pending，失败时由 resetResult 清掉旧结果与热力图 */
    markResultPending('正在识别照片…');
    recognizeSnapshot(dataUrl, false);
  }

  /* 当前仍在展示的拍照快照（dataURL）；停止摄像头或继续实时后会被清空 */
  function currentCameraSnapshot() {
    if (!D.cameraSnapshot || D.cameraSnapshot.hidden) {
      return '';
    }
    var src = D.cameraSnapshot.getAttribute('src') || '';
    return /^data:image\//.test(src) ? src : '';
  }

  /* 用当前模型识别快照。reRecognize=true 表示由模型/模型组切换触发。 */
  function recognizeSnapshot(dataUrl, reRecognize, reason) {
    if (!state.modelsReady || !state.model) {
      return;
    }

    state.camera.photoBusy = true;
    updateCameraControls();
    if (reRecognize) {
      setCameraStatus(reason === 'gradcam'
        ? '正在重新生成 Grad-CAM…'
        : '正在用新模型重新识别照片…', '');
    }

    /* Grad-CAM 开关状态在发请求时读取，拍照/重识别都不再强制请求热力图 */
    var request = sendFrame(dataUrl);
    var seq = request.seq;
    state.camera.photoSeq = seq;

    request.then(function () {
      /* 只有最新一次快照请求负责收尾（实时帧会推进 cameraSeq，但不影响 photoSeq） */
      if (seq !== state.camera.photoSeq) {
        return;
      }
      state.camera.photoBusy = false;
      updateCameraControls();
      /* 请求已被停止摄像头/切页作废：不要留下过期的 pending 标记 */
      if (seq !== state.cameraSeq) {
        if (resultIsPending() && !state.camera.running) {
          resetResult('重识别已取消，请重新拍照或上传图片');
        }
        return;
      }
      if (state.camera.running && state.camera.paused &&
          state.activeSection === 'recognize' && cameraPanelActive()) {
        setCameraStatus('已拍照并暂停实时识别，点击「继续」恢复', '');
      }
    }, function (err) {
      if (seq !== state.camera.photoSeq) {
        return;
      }
      state.camera.photoBusy = false;
      updateCameraControls();
      if (err && err.__cameraStale) {
        /* 请求已被停止摄像头/切页作废：不要留下过期的 pending 标记 */
        if (resultIsPending() && !state.camera.running) {
          resetResult('拍照识别已取消，请重新拍照或上传图片');
        }
        return;
      }
      var message = '拍照识别失败：' + err.message;
      toast(message, 'error', 5200);
      /* 重识别/新拍照失败时不要继续把旧结果当作新结果展示 */
      if (D.resultBody && !D.resultBody.hidden) {
        if (reason === 'gradcam') {
          /* 只是热力图请求失败：保留识别结果，不把整块结果清空 */
          if (gradcamEnabled()) {
            showGradcamUnavailable('Grad-CAM 生成失败，请稍后重试');
          } else {
            hideGradcam();
          }
        } else if (reRecognize) {
          resetResult('重新识别失败：' + err.message + '，请重新拍照或稍后重试');
        } else {
          resetResult('拍照识别失败：' + err.message + '，请重新拍照或稍后重试');
        }
      }
      /* 失败时状态栏显示真实错误，不写“已拍照并暂停…”的成功文案 */
      if (state.camera.running && state.camera.paused &&
          state.activeSection === 'recognize' && cameraPanelActive()) {
        setCameraStatus(message, 'error');
      }
    });
  }

  function isCameraRequestFresh(seq) {
    /* 上传请求在途不影响摄像头结果：两者各有独立的来源与序号 */
    return seq === state.cameraSeq &&
      state.camera.running &&
      state.activeSection === 'recognize' &&
      cameraPanelActive();
  }

  function sendFrame(dataUrl) {
    var seq = ++state.cameraSeq;
    /* 热力图是否请求完全由开关决定：拍照快照与实时帧共用同一规则 */
    var wantGradcam = gradcamEnabled();
    var payload = {
      model: state.model ? state.model.id : '',
      group: state.group ? state.group.id : '',
      image_data_url: dataUrl,
      gradcam: wantGradcam,
      topk: topkForGroup()
    };

    var request = postJSON('/api/predict_frame', payload).then(function (data) {
      var fresh = isCameraRequestFresh(seq);
      if (fresh) {
        renderResult(data, wantGradcam);
      }
      return { data: data, fresh: fresh };
    }, function (err) {
      var stale = !isCameraRequestFresh(seq);
      if (err && typeof err === 'object') {
        err.__cameraStale = stale;
      }
      throw err;
    });

    /* 调用方用它判断自己的请求是否已被后续请求取代 */
    request.seq = seq;
    return request;
  }

  /* ============================================================
     上传图片
     ============================================================ */

  function bindUpload() {
    if (!D.dropzone) {
      return;
    }

    D.dropzone.addEventListener('click', function () {
      if (!state.uploading) {
        D.fileInput.click();
      }
    });

    D.dropzone.addEventListener('keydown', function (event) {
      if (event.key === 'Enter' || event.key === ' ' || event.key === 'Spacebar') {
        event.preventDefault();
        if (!state.uploading) {
          D.fileInput.click();
        }
      }
    });

    D.fileInput.addEventListener('change', function () {
      var file = this.files && this.files[0];
      if (file) {
        handleUpload(file);
      }
      this.value = '';
    });

    ['dragenter', 'dragover'].forEach(function (type) {
      D.dropzone.addEventListener(type, function (event) {
        event.preventDefault();
        event.stopPropagation();
        if (!state.uploading) {
          D.dropzone.classList.add('is-dragover');
        }
      });
    });

    ['dragleave', 'dragend'].forEach(function (type) {
      D.dropzone.addEventListener(type, function (event) {
        event.preventDefault();
        event.stopPropagation();
        D.dropzone.classList.remove('is-dragover');
      });
    });

    D.dropzone.addEventListener('drop', function (event) {
      event.preventDefault();
      event.stopPropagation();
      D.dropzone.classList.remove('is-dragover');
      if (state.uploading) {
        return;
      }
      var transfer = event.dataTransfer;
      var file = transfer && transfer.files && transfer.files[0];
      if (file) {
        handleUpload(file);
      }
    });

    if (D.clearUploadBtn) {
      D.clearUploadBtn.addEventListener('click', clearUpload);
    }
  }

  function handleUpload(file) {
    var problem = validateImage(file);
    if (problem) {
      toast(problem, 'error', 4800);
      return;
    }

    state.uploadFile = file;

    if (state.uploadPreviewUrl) {
      URL.revokeObjectURL(state.uploadPreviewUrl);
    }
    state.uploadPreviewUrl = URL.createObjectURL(file);
    D.uploadPreview.src = state.uploadPreviewUrl;
    D.uploadPreviewWrap.hidden = false;
    D.uploadFileName.textContent = file.name + ' · ' + formatBytes(file.size);

    if (!ensureModelReady(false)) {
      setUploadStatus('模型尚未就绪，图片已预览，可稍后重新选择', 'error');
      resetResult('模型尚未就绪，图片未识别，请稍后重新选择');
      return;
    }

    /* 新上传开始即进入 pending，避免请求失败时残留上一张图的结果与热力图 */
    markResultPending('正在识别图片…');
    predictFile(file);
  }

  function validateImage(file) {
    var name = String(file.name || '').toLowerCase();
    var typeOk = file.type === 'image/jpeg' || file.type === 'image/png' ||
      /\.(jpe?g|png)$/.test(name);
    if (!typeOk) {
      return '仅支持 JPG / PNG 图片';
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      return '图片超过 10 MB 限制，请压缩后重试';
    }
    return '';
  }

  /* 响应对应的模型/模型组是否仍是当前选择（配合 uploadSeq 防止迟到响应覆盖） */
  function uploadRequestIsCurrent(groupId, modelId) {
    return Boolean(state.group && state.model &&
      state.group.id === groupId && state.model.id === modelId);
  }

  function predictFile(file, reRecognize, reason) {
    state.uploading = true;
    setUploadBusy(true);
    setUploadStatus(reRecognize
      ? (reason === 'gradcam'
        ? '正在重新生成 Grad-CAM…'
        : '正在用新模型重新识别…')
      : '正在识别：' + file.name + ' …', '');

    var seq = ++state.uploadSeq;
    var wantGradcam = gradcamEnabled();
    var requestGroupId = state.group.id;
    var requestModelId = state.model.id;
    var form = new FormData();
    form.append('image', file, file.name || 'upload.png');
    form.append('model', requestModelId);
    form.append('group', requestGroupId);
    form.append('topk', String(topkForGroup()));
    form.append('gradcam', wantGradcam ? '1' : '0');

    return fetchJSON('/api/predict', {
      method: 'POST',
      body: form
    }).then(function (data) {
      /* 过期请求的结果不得覆盖新模型的结果 */
      if (seq !== state.uploadSeq) {
        return;
      }
      /* 请求归属校验：请求发出后模型/模型组已变化（含切到无可用模型的组）时，
         迟到响应不得再用旧组/旧模型结果覆盖当前界面 */
      if (!uploadRequestIsCurrent(requestGroupId, requestModelId)) {
        return;
      }
      /* 上传面板已被摄像头取代时，只更新上传页状态，不写共享结果区 */
      if (cameraPanelActive()) {
        setUploadStatus('识别完成 · ' + file.name + '（已切换到摄像头，结果未覆盖显示）', '');
        if (resultIsPending() && !state.camera.running && !state.camera.starting &&
            !currentCameraSnapshot()) {
          resetResult('上传结果未覆盖显示；返回「上传图片」可重新识别');
        }
        return;
      }
      renderResult(data, wantGradcam);
      setUploadStatus('识别完成 · ' + file.name, '');
    }).catch(function (err) {
      if (seq !== state.uploadSeq) {
        return;
      }
      if (!uploadRequestIsCurrent(requestGroupId, requestModelId)) {
        return;
      }
      setUploadStatus('识别失败：' + err.message, 'error');
      if (cameraPanelActive()) {
        if (resultIsPending() && !state.camera.running && !state.camera.starting &&
            !currentCameraSnapshot()) {
          resetResult('上传重识别失败，已切换到摄像头；返回「上传图片」可重试');
        }
        return;
      }
      toast('识别失败：' + err.message, 'error', 5600);
      /* 新上传/重识别失败时不要继续展示旧结果与旧热力图 */
      if (D.resultBody && !D.resultBody.hidden) {
        if (reason === 'gradcam') {
          /* 只是热力图请求失败：保留识别结果，不清空整块结果 */
          if (gradcamEnabled()) {
            showGradcamUnavailable('Grad-CAM 生成失败，请稍后重试');
          } else {
            hideGradcam();
          }
        } else if (reRecognize) {
          resetResult('重新识别失败：' + err.message + '，请重新选择图片或稍后重试');
        } else {
          resetResult('识别失败：' + err.message + '，请重新选择图片或稍后重试');
        }
      }
    }).then(function () {
      if (seq === state.uploadSeq) {
        state.uploading = false;
        setUploadBusy(false);
      }
    });
  }

  function setUploadBusy(busy) {
    if (D.dropzone) {
      D.dropzone.classList.toggle('is-busy', busy);
    }
    if (D.fileInput) {
      D.fileInput.disabled = busy;
    }
    if (D.clearUploadBtn) {
      D.clearUploadBtn.disabled = busy;
    }
  }

  function setUploadStatus(text, kind) {
    if (!D.uploadStatus) {
      return;
    }
    D.uploadStatus.textContent = text;
    D.uploadStatus.className = 'camera-status';
    if (kind === 'error') {
      D.uploadStatus.classList.add('is-error');
    } else if (kind === 'live') {
      D.uploadStatus.classList.add('is-live');
    }
  }

  function clearUpload() {
    if (state.uploading) {
      return;
    }
    state.uploadFile = null;
    if (state.uploadPreviewUrl) {
      URL.revokeObjectURL(state.uploadPreviewUrl);
      state.uploadPreviewUrl = null;
    }
    D.uploadPreview.removeAttribute('src');
    D.uploadPreviewWrap.hidden = true;
    D.uploadFileName.textContent = '—';
    setUploadStatus('选择图片后将自动识别', '');
    resetResult('选择图片后将自动识别');
  }

  /* ============================================================
     Grad-CAM 开关
     ============================================================ */

  function bindGradcam() {
    if (!D.gradcamToggle) {
      return;
    }
    D.gradcamToggle.addEventListener('change', function () {
      if (!this.checked) {
        /* 关闭：立即隐藏并清掉旧热力图，避免在任何后续响应里残留 */
        hideGradcam();
        return;
      }
      refreshGradcamForCurrent();
    });

    /* 后端返回的 base64 若无法解码，降级为提示而不是破图 */
    if (D.gradcamImage) {
      D.gradcamImage.addEventListener('error', function () {
        if (!gradcamEnabled()) {
          return;
        }
        showGradcamUnavailable('Grad-CAM 暂不可用');
      });
    }
  }

  function gradcamEnabled() {
    return Boolean(D.gradcamToggle && D.gradcamToggle.checked);
  }

  function hideGradcam() {
    state.gradcamPending = false;
    if (D.gradcamPanel) {
      D.gradcamPanel.hidden = true;
    }
    if (D.gradcamImage) {
      D.gradcamImage.hidden = true;
      D.gradcamImage.removeAttribute('src');
    }
    if (D.gradcamUnavailable) {
      D.gradcamUnavailable.hidden = true;
    }
  }

  function showGradcamUnavailable(message) {
    if (!gradcamEnabled()) {
      hideGradcam();
      return;
    }
    state.gradcamPending = false;
    if (D.gradcamPanel) {
      D.gradcamPanel.hidden = false;
    }
    if (D.gradcamImage) {
      D.gradcamImage.hidden = true;
      D.gradcamImage.removeAttribute('src');
    }
    if (D.gradcamUnavailable) {
      D.gradcamUnavailable.textContent = message || 'Grad-CAM 暂不可用';
      D.gradcamUnavailable.hidden = false;
    }
  }

  /* 打开开关后为正在生成的请求占位；没有识别结果时不显示任何热力图 */
  function showGradcamPending() {
    if (!gradcamEnabled() || !D.resultBody || D.resultBody.hidden) {
      return;
    }
    showGradcamUnavailable('正在生成 Grad-CAM…');
    state.gradcamPending = true;
  }

  /* 停止摄像头/离开识别页后，pending 占位不会再有响应来更新，主动收尾 */
  function settleGradcamAfterCameraStop() {
    if (state.gradcamPending) {
      hideGradcam();
    }
  }

  /* 开关打开：有上传图片/拍照快照则重新请求；实时交给下一帧；
     没有任何素材时只提示、不发请求、不伪造热力图。 */
  function refreshGradcamForCurrent() {
    if (!gradcamEnabled()) {
      hideGradcam();
      return;
    }
    if (reRecognizeCurrent('gradcam')) {
      return;
    }
    if (D.resultBody && !D.resultBody.hidden) {
      showGradcamUnavailable(cameraPanelActive() && state.uploadFile
        ? '请切回「上传图片」后重新识别'
        : '暂无可解释的图片');
    } else {
      hideGradcam();
    }
  }

  function showGradcam(base64) {
    if (!gradcamEnabled()) {
      hideGradcam();
      return;
    }
    state.gradcamPending = false;
    D.gradcamPanel.hidden = false;
    D.gradcamImage.hidden = false;
    D.gradcamUnavailable.hidden = true;
    var source = /^data:image\//.test(base64) ? base64 : 'data:image/png;base64,' + base64;
    D.gradcamImage.src = source;
  }

  /* ============================================================
     识别结果渲染
     ============================================================ */

  function renderResult(data, requestGradcam) {
    if (!data || !Array.isArray(data.topk) || !data.topk.length) {
      toast('识别接口返回了空结果', 'error', 4800);
      return;
    }

    state.lastResult = data;
    state.resultPending = false;
    D.resultEmpty.hidden = true;
    D.resultBody.hidden = false;

    var top = data.topk[0];
    var topProb = Number(top.prob);
    if (!isFinite(topProb)) {
      topProb = 0;
    }

    D.resultZh.textContent = top.zh_name || top.key || '未知类别';
    D.resultSci.textContent = top.scientific_name || top.scientific || '';
    D.resultLatency.textContent = isFinite(Number(data.latency_ms))
      ? Math.round(Number(data.latency_ms)) + ' ms'
      : '—';

    var tagParts = [];
    if (data.model) {
      tagParts.push('模型 ' + (data.model_name || data.model));
    }
    if (data.group || data.group_label) {
      tagParts.push('模型组 ' + (data.group_label || data.group));
    }
    D.resultModelTag.textContent = tagParts.join(' · ');
    D.resultModelTag.hidden = !tagParts.length;

    var isLow = topProb < 0.5;
    D.lowConfidenceWarning.hidden = !isLow;
    if (isLow) {
      D.lowConfidenceWarning.textContent =
        '置信度仅 ' + formatPct(topProb) + '（低于 50%），结果可能不可靠，请调整叶片位置或更换图片后重试。';
    }

    renderTopk(data.topk);

    /* 渲染以当前开关为准：关闭时无论响应是否带 base64 都不显示热力图 */
    if (!gradcamEnabled()) {
      hideGradcam();
    } else if (data.gradcam_available && data.gradcam_png_base64) {
      showGradcam(data.gradcam_png_base64);
    } else if (requestGradcam === false) {
      /* 该响应对应的是关闭开关时的请求，等下一次带 gradcam 的响应 */
      showGradcamUnavailable('正在生成 Grad-CAM…');
    } else {
      /* 后端明确不可用/生成失败：给通用提示，不暴露异常堆栈，也不显示破图 */
      showGradcamUnavailable(data.gradcam_error
        ? 'Grad-CAM 生成失败，请稍后重试'
        : 'Grad-CAM 暂不可用');
    }
  }

  function renderTopk(topk) {
    D.topkList.textContent = '';

    topk.forEach(function (item, index) {
      var probability = Number(item.prob);
      if (!isFinite(probability)) {
        probability = 0;
      }
      var percent = Math.max(0, Math.min(100, probability * 100));

      var li = document.createElement('li');
      li.className = 'topk-item' + (index === 0 ? ' is-top' : '');

      var trackWrap = document.createElement('div');
      trackWrap.className = 'topk-track-wrap';

      var label = document.createElement('div');
      label.className = 'topk-label';

      var zh = document.createElement('strong');
      zh.textContent = item.zh_name || item.key || '未知';
      label.appendChild(zh);

      var sci = document.createElement('span');
      sci.textContent = item.scientific_name || item.scientific || item.key || '';
      label.appendChild(sci);

      var track = document.createElement('div');
      track.className = 'topk-track';
      var bar = document.createElement('span');
      bar.className = 'topk-bar';
      bar.style.width = percent.toFixed(1) + '%';
      track.appendChild(bar);

      trackWrap.appendChild(label);
      trackWrap.appendChild(track);

      var value = document.createElement('span');
      value.className = 'topk-prob';
      value.textContent = formatPct(probability);

      li.appendChild(trackWrap);
      li.appendChild(value);
      D.topkList.appendChild(li);
    });
  }

  /* 重识别进行中：在旧结果的模型标签上明确标注，避免误认为已是新模型结果 */
  function markResultPending(message) {
    if (!D.resultModelTag || !D.resultBody || D.resultBody.hidden) {
      return;
    }
    state.resultPending = true;
    D.resultModelTag.textContent = message || '正在用新模型重新识别…';
    D.resultModelTag.hidden = false;
    /* 新结果返回前不保留旧结果的热力图 */
    if (gradcamEnabled()) {
      showGradcamPending();
    }
  }

  function resultIsPending() {
    return state.resultPending;
  }

  function resetResult(message) {
    state.lastResult = null;
    state.resultPending = false;
    D.resultEmpty.hidden = false;
    D.resultBody.hidden = true;
    hideGradcam();

    var small = D.resultEmpty.querySelector('small');
    if (small) {
      small.textContent = message || '开启摄像头或上传一张叶片图片开始识别';
    }
  }

  function ensureModelReady(silent) {
    if (state.modelsReady && state.group && state.model) {
      return true;
    }
    if (!silent) {
      if (!state.modelsReady) {
        toast('模型尚未加载完成，请稍候重试', 'warn', 4200);
      } else {
        toast('当前模型组没有可用模型', 'warn', 4200);
      }
    }
    return false;
  }

  /* ============================================================
     系统操作演示
     ============================================================ */

  function bindDemo() {
    if (D.addDataBtn) {
      D.addDataBtn.addEventListener('click', handleAddData);
    }
    if (D.trainBtn) {
      D.trainBtn.addEventListener('click', handleTrain);
    }
  }

  function handleAddData() {
    toast('演示功能，未接入真实数据接口', 'info', 4400);

    if (state.demoBusy) {
      return;
    }
    state.demoBusy = true;
    setButtonBusy(D.addDataBtn, true);

    postJSON('/api/demo/add-data', {}).catch(function (err) {
      toast('演示接口未响应：' + err.message, 'error', 4400);
    }).then(function () {
      state.demoBusy = false;
      setButtonBusy(D.addDataBtn, false);
    });
  }

  function handleTrain() {
    if (state.demoBusy) {
      return;
    }
    state.demoBusy = true;

    openConfirm({
      title: '开始模型训练？',
      text: '这是演示功能，不会启动真实训练。确认后仅模拟一次约 2–3 秒的训练进度。',
      confirmText: '开始演示',
      cancelText: '取消'
    }).then(function (confirmed) {
      if (!confirmed) {
        state.demoBusy = false;
        return;
      }
      runDemoTraining();
    });
  }

  function runDemoTraining() {
    setButtonBusy(D.trainBtn, true);

    var apiPromise = postJSON('/api/demo/train', {}).catch(function (err) {
      toast('演示接口未响应：' + err.message, 'error', 4400);
    });

    enterProgressMode();

    var totalMs = 2400;
    var started = Date.now();

    var progressPromise = new Promise(function (resolve) {
      var timer = setInterval(function () {
        var elapsed = Date.now() - started;
        var percent = Math.min(100, Math.round((elapsed / totalMs) * 100));
        setProgress(percent);
        if (percent >= 100) {
          clearInterval(timer);
          resolve();
        }
      }, 80);
    });

    progressPromise.then(function () {
      return apiPromise;
    }).then(function () {
      setProgress(100);
      return sleep(280);
    }).then(function () {
      setButtonBusy(D.trainBtn, false);
      closeModal();
      state.demoBusy = false;
      toast('演示功能，未启动真实训练', 'info', 4600);
    });
  }

  function setButtonBusy(button, busy) {
    if (!button) {
      return;
    }
    button.classList.toggle('is-busy', busy);
    button.disabled = busy;
  }

  /* ============================================================
     确认 / 进度弹窗
     ============================================================ */

  function bindModal() {
    if (!D.modalBackdrop) {
      return;
    }

    D.modalConfirm.addEventListener('click', confirmModal);
    D.modalCancel.addEventListener('click', closeModal);

    D.modalBackdrop.addEventListener('click', function (event) {
      if (event.target === D.modalBackdrop && state.modal.mode === 'confirm') {
        closeModal();
      }
    });

    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape' && !D.modalBackdrop.hidden && state.modal.mode === 'confirm') {
        closeModal();
      }
      if (event.key === 'Tab' && !D.modalBackdrop.hidden && state.modal.mode === 'confirm') {
        trapModalFocus(event);
      }
    });
  }

  function openConfirm(options) {
    return new Promise(function (resolve) {
      state.modal.resolve = resolve;
      state.modal.mode = 'confirm';
      state.modal.trigger = document.activeElement;

      D.modalTitle.textContent = options.title || '确认操作';
      D.modalText.textContent = options.text || '';
      D.modalConfirm.textContent = options.confirmText || '确认';
      D.modalCancel.textContent = options.cancelText || '取消';
      D.modalCancel.hidden = false;
      D.modalActions.hidden = false;
      D.trainProgressWrap.hidden = true;
      D.trainProgressBar.style.width = '0%';
      D.trainProgressText.textContent = '0%';
      D.modalBackdrop.hidden = false;

      document.body.classList.add('is-locked');
      window.setTimeout(function () {
        D.modalConfirm.focus();
      }, 0);
    });
  }

  function confirmModal() {
    var resolve = state.modal.resolve;
    state.modal.resolve = null;
    D.modalBackdrop.hidden = true;
    document.body.classList.remove('is-locked');
    if (typeof resolve === 'function') {
      resolve(true);
    }
  }

  function closeModal() {
    var resolve = state.modal.resolve;
    var mode = state.modal.mode;
    var trigger = state.modal.trigger;
    state.modal.resolve = null;
    state.modal.mode = 'confirm';
    state.modal.trigger = null;

    D.modalBackdrop.hidden = true;
    document.body.classList.remove('is-locked');
    D.modalActions.hidden = false;
    D.trainProgressWrap.hidden = true;

    if (mode === 'confirm' && typeof resolve === 'function') {
      resolve(false);
    }

    if (trigger && trigger !== document.body &&
        typeof trigger.focus === 'function' &&
        document.contains(trigger)) {
      trigger.focus();
    }
  }

  function enterProgressMode() {
    state.modal.mode = 'progress';
    state.modal.resolve = null;

    D.modalTitle.textContent = '模型训练（演示）';
    D.modalText.textContent = '正在模拟训练流程，本操作不会启动真实训练进程。';
    D.trainProgressWrap.hidden = false;
    D.modalActions.hidden = true;
    D.modalBackdrop.hidden = false;
    document.body.classList.add('is-locked');
    setProgress(0);

    D.modalTitle.setAttribute('tabindex', '-1');
    window.setTimeout(function () {
      D.modalTitle.focus();
    }, 0);
  }

  function setProgress(percent) {
    var value = Math.max(0, Math.min(100, Math.round(Number(percent) || 0)));
    D.trainProgressBar.style.width = value + '%';
    D.trainProgressText.textContent = value + '%';

    var track = D.trainProgressWrap.querySelector('.progress-track');
    if (track) {
      track.setAttribute('aria-valuenow', String(value));
    }
  }

  function trapModalFocus(event) {
    var focusable = D.modalBackdrop.querySelectorAll('button:not([disabled]), [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
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
     网络请求
     ============================================================ */

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

  function postJSON(url, payload) {
    return fetchJSON(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload || {})
    });
  }

  /* ============================================================
     工具函数
     ============================================================ */

  function toast(message, type, duration) {
    if (!D.toastContainer) {
      return;
    }
    var kind = type || 'info';
    var element = document.createElement('div');
    element.className = 'toast toast-' + kind;
    element.setAttribute('role', kind === 'error' ? 'alert' : 'status');

    var text = document.createElement('span');
    text.textContent = message;
    element.appendChild(text);

    var close = document.createElement('button');
    close.type = 'button';
    close.className = 'toast-close';
    close.setAttribute('aria-label', '关闭提示');
    close.textContent = '×';
    close.addEventListener('click', function () {
      element.remove();
    });
    element.appendChild(close);

    D.toastContainer.appendChild(element);

    window.requestAnimationFrame(function () {
      element.classList.add('is-visible');
    });

    window.setTimeout(function () {
      element.classList.remove('is-visible');
      window.setTimeout(function () {
        if (element.parentNode) {
          element.remove();
        }
      }, 220);
    }, duration || 3800);
  }

  function formatPct(value) {
    var number = Number(value);
    if (!isFinite(number)) {
      return '—';
    }
    return (number * 100).toFixed(1) + '%';
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

  function sleep(ms) {
    return new Promise(function (resolve) {
      window.setTimeout(resolve, ms);
    });
  }

  /* ---------- 暴露给 gallery.js / 调试 ---------- */

  window.LeafApp = {
    toast: toast,
    state: state,
    activateSection: activateSection,
    stopCamera: stopCamera
  };
})();
