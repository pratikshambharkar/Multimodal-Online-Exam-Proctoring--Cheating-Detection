/**
 * RAIM — client-side proctoring agent (vanilla ES6, no dependencies).
 *
 * Usage (from exam_take.html):
 *   Proctor.init({
 *     sessionId: 12,
 *     frameUrl: '/proctoring/api/frame/',
 *     audioUrl: '/proctoring/api/audio/',
 *     eventUrl: '/proctoring/api/event/',
 *     statusUrl: '/proctoring/api/status/12/',
 *     frameIntervalMs: 2500,
 *     audioChunkMs: 10000,
 *     submitFormId: 'exam-form'
 *   });
 *
 * Captures webcam frames + microphone audio, streams them to the server,
 * reports browser events (tab switch, blur, fullscreen exit, copy/paste,
 * right click), renders the live monitoring panel into #proctor-panel and
 * auto-submits the exam form when the server orders termination.
 */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ */
  /* Helpers                                                             */
  /* ------------------------------------------------------------------ */

  function getCookie(name) {
    var value = null;
    if (document.cookie && document.cookie !== '') {
      var cookies = document.cookie.split(';');
      for (var i = 0; i < cookies.length; i++) {
        var cookie = cookies[i].trim();
        if (cookie.substring(0, name.length + 1) === name + '=') {
          value = decodeURIComponent(cookie.substring(name.length + 1));
          break;
        }
      }
    }
    return value;
  }

  /** Wrap a handler so it can never throw uncaught. */
  function safe(fn) {
    return function () {
      try {
        return fn.apply(this, arguments);
      } catch (err) {
        if (window.console && console.error) {
          console.error('[Proctor]', err);
        }
      }
    };
  }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function timeStamp() {
    var d = new Date();
    function p(n) { return (n < 10 ? '0' : '') + n; }
    return p(d.getHours()) + ':' + p(d.getMinutes()) + ':' + p(d.getSeconds());
  }

  /** Average-block downsample of Float32 PCM to a lower sample rate. */
  function downsample(buffer, srcRate, dstRate) {
    if (dstRate >= srcRate) return buffer;
    var ratio = srcRate / dstRate;
    var newLen = Math.floor(buffer.length / ratio);
    var result = new Float32Array(newLen);
    var offsetBuffer = 0;
    for (var i = 0; i < newLen; i++) {
      var nextOffset = Math.round((i + 1) * ratio);
      var accum = 0;
      var count = 0;
      for (var j = offsetBuffer; j < nextOffset && j < buffer.length; j++) {
        accum += buffer[j];
        count++;
      }
      result[i] = count > 0 ? accum / count : 0;
      offsetBuffer = nextOffset;
    }
    return result;
  }

  /** Encode mono Float32 samples as a 16-bit PCM WAV blob (RIFF header). */
  function encodeWav(samples, sampleRate) {
    var buffer = new ArrayBuffer(44 + samples.length * 2);
    var view = new DataView(buffer);

    function writeString(offset, str) {
      for (var i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i));
      }
    }

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);                 // fmt chunk size
    view.setUint16(20, 1, true);                  // PCM
    view.setUint16(22, 1, true);                  // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);     // byte rate
    view.setUint16(32, 2, true);                  // block align
    view.setUint16(34, 16, true);                 // bits per sample
    writeString(36, 'data');
    view.setUint32(40, samples.length * 2, true);

    var offset = 44;
    for (var i = 0; i < samples.length; i++, offset += 2) {
      var s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return new Blob([view], { type: 'audio/wav' });
  }

  /* ------------------------------------------------------------------ */
  /* Proctor singleton                                                   */
  /* ------------------------------------------------------------------ */

  var Proctor = {
    opts: null,
    panel: null,
    video: null,
    form: null,
    overlay: null,
    canvas: null,

    active: false,
    terminated: false,
    frameInFlight: false,
    connected: true,

    stream: null,
    frameTimer: null,
    audioTimer: null,
    meterTimer: null,

    audioCtx: null,
    audioSource: null,
    audioProcessor: null,
    audioSilencer: null,
    audioChunks: [],
    audioSampleRate: 48000,
    audioLevel: 0,

    identityState: null,      // null = pending, true = verified, false = failed
    tabCount: 0,
    lastHiddenAt: 0,
    blurTimer: null,
    eventsBound: false,
    handlers: null,

    ui: null,

    /* ------------------------------ setup ------------------------------ */

    init: function (opts) {
      try {
        Proctor.opts = Object.assign(
          { frameIntervalMs: 2500, audioChunkMs: 10000 },
          opts || {}
        );
        Proctor.panel = document.getElementById('proctor-panel');
        Proctor.video = document.getElementById('proctor-video');
        Proctor.form = Proctor.opts.submitFormId
          ? document.getElementById(Proctor.opts.submitFormId)
          : null;
        if (!Proctor.panel) return;

        Proctor.canvas = document.createElement('canvas');
        Proctor.canvas.width = 640;
        Proctor.canvas.height = 480;

        Proctor.buildPanel();
        Proctor.buildOverlay();

        window.addEventListener('beforeunload', Proctor._onBeforeUnload);
        document.addEventListener('click', safe(Proctor._resumeAudioCtx));
        if (Proctor.form) {
          Proctor.form.addEventListener('submit', safe(function () {
            Proctor.stop();
          }));
        }
      } catch (err) {
        if (window.console && console.error) console.error('[Proctor] init failed', err);
      }
    },

    _onBeforeUnload: function (e) {
      if (Proctor.active && !Proctor.terminated) {
        e.preventDefault();
        e.returnValue = 'Your proctored exam is in progress. Leaving now will be recorded.';
        return e.returnValue;
      }
    },

    _resumeAudioCtx: function () {
      if (Proctor.audioCtx && Proctor.audioCtx.state === 'suspended') {
        Proctor.audioCtx.resume().catch(function () {});
      }
    },

    /* --------------------------- panel DOM ----------------------------- */

    buildPanel: function () {
      var p = Proctor.panel;
      p.innerHTML = '';
      Proctor.ui = { chips: {} };

      // Webcam preview: reuse #proctor-video if the template placed one,
      // otherwise create it inside the panel.
      if (!Proctor.video) {
        Proctor.video = document.createElement('video');
        Proctor.video.id = 'proctor-video';
      }
      Proctor.video.autoplay = true;
      Proctor.video.muted = true;
      Proctor.video.setAttribute('playsinline', '');
      if (!Proctor.video.parentNode) {
        var wrap = el('div', 'proctor-video-wrap');
        wrap.appendChild(Proctor.video);
        p.appendChild(wrap);
      }

      var head = el('div', 'proctor-head');
      head.appendChild(el('strong', null, 'Live Monitor'));
      Proctor.ui.dot = el('span', 'proctor-dot');
      head.appendChild(Proctor.ui.dot);
      p.appendChild(head);

      Proctor.ui.banner = el('div', 'proctor-banner bad',
        'No offline mode — reconnect immediately');
      Proctor.ui.banner.hidden = true;
      p.appendChild(Proctor.ui.banner);

      // Risk meter
      var meter = el('div', 'risk-meter');
      var top = el('div', 'risk-meter-top');
      top.appendChild(el('span', 'risk-label', 'Risk score'));
      Proctor.ui.riskValue = el('span', 'risk-value', '0%');
      top.appendChild(Proctor.ui.riskValue);
      meter.appendChild(top);
      var bar = el('div', 'risk-bar');
      Proctor.ui.riskFill = el('div', 'risk-fill risk-green');
      Proctor.ui.riskFill.style.width = '0%';
      bar.appendChild(Proctor.ui.riskFill);
      meter.appendChild(bar);
      p.appendChild(meter);

      // Status chips
      var chipRow = el('div', 'chip-row');
      [
        ['face', 'Face', '—'],
        ['identity', 'Identity', 'Pending'],
        ['gaze', 'Gaze', '—'],
        ['objects', 'Objects', '—'],
        ['audio', 'Audio', '—'],
        ['tabs', 'Tabs', '0']
      ].forEach(function (spec) {
        var chip = el('div', 'chip');
        chip.appendChild(el('span', 'chip-name', spec[1]));
        var value = el('span', 'chip-value', spec[2]);
        chip.appendChild(value);
        chipRow.appendChild(chip);
        Proctor.ui.chips[spec[0]] = { root: chip, value: value };
      });
      p.appendChild(chipRow);

      // Mic level meter
      var audioMeter = el('div', 'audio-meter');
      audioMeter.appendChild(el('span', 'audio-meter-label', 'Mic'));
      var audioBar = el('div', 'audio-bar');
      Proctor.ui.audioFill = el('div', 'audio-fill');
      audioBar.appendChild(Proctor.ui.audioFill);
      audioMeter.appendChild(audioBar);
      p.appendChild(audioMeter);

      // Warnings feed
      p.appendChild(el('div', 'proctor-warnings-title', 'Alerts'));
      Proctor.ui.warnings = el('ul', 'proctor-warnings');
      Proctor.ui.warnings.appendChild(el('li', 'empty', 'No alerts yet.'));
      p.appendChild(Proctor.ui.warnings);
    },

    buildOverlay: function () {
      var overlay = el('div', 'proctor-overlay');
      overlay.id = 'proctor-overlay';
      overlay.appendChild(Proctor._overlayStartCard());
      document.body.appendChild(overlay);
      Proctor.overlay = overlay;
    },

    _overlayStartCard: function () {
      var card = el('div', 'overlay-card');
      card.appendChild(el('h2', null, 'Proctored Exam'));
      card.appendChild(el('p', null,
        'This exam is continuously monitored by RAIM. Before you begin, ' +
        'monitoring must be activated.'));
      var list = el('ul', 'overlay-list');
      [
        'Your webcam and microphone stay on for the whole exam.',
        'The exam runs in fullscreen; leaving it is recorded.',
        'Tab switches, window changes, copy/paste and right-click are logged.',
        'AI analysis flags absent or extra faces, phones, books and speech.'
      ].forEach(function (text) {
        list.appendChild(el('li', null, text));
      });
      card.appendChild(list);

      var btn = el('button', 'btn btn-primary btn-lg', 'Start monitoring');
      btn.type = 'button';
      btn.addEventListener('click', safe(function () {
        btn.disabled = true;
        btn.textContent = 'Requesting permissions…';
        Proctor.begin();
      }));
      card.appendChild(btn);
      card.appendChild(el('p', 'overlay-note',
        'By starting, you consent to camera, microphone and browser-activity ' +
        'monitoring for the duration of this exam.'));
      return card;
    },

    removeOverlay: function () {
      if (Proctor.overlay) {
        Proctor.overlay.remove();
        Proctor.overlay = null;
      }
    },

    /* ------------------------- start / media --------------------------- */

    begin: function () {
      // User gesture: request fullscreen politely, then media.
      try {
        var docEl = document.documentElement;
        if (docEl.requestFullscreen) {
          var fsPromise = docEl.requestFullscreen();
          if (fsPromise && fsPromise.catch) fsPromise.catch(function () {});
        }
      } catch (err) { /* fullscreen refusal is not fatal */ }
      Proctor.startMedia();
    },

    startMedia: function () {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        Proctor.showMediaError(
          'This browser cannot capture camera and microphone. ' +
          'Please use an up-to-date version of Chrome or Edge.');
        return;
      }
      navigator.mediaDevices
        .getUserMedia({ video: { width: 640, height: 480 }, audio: true })
        .then(safe(function (stream) {
          Proctor.stream = stream;
          Proctor.video.srcObject = stream;
          var playPromise = Proctor.video.play();
          if (playPromise && playPromise.catch) playPromise.catch(function () {});
          Proctor.startAudio(stream);
          Proctor.bindBrowserEvents();
          Proctor.startLoops();
          Proctor.fetchStatus();
          Proctor.removeOverlay();
          Proctor.active = true;
          Proctor.setConnected(true);
        }))
        .catch(safe(function () {
          Proctor.showMediaError(
            'Camera and microphone access are mandatory for this proctored ' +
            'exam. Please allow both permissions in your browser and retry.');
        }));
    },

    showMediaError: function (message) {
      function errorCard() {
        var card = el('div', 'proctor-error');
        card.appendChild(el('h3', null, 'Monitoring blocked'));
        card.appendChild(el('p', null, message));
        var retry = el('button', 'btn btn-danger', 'Retry camera & microphone');
        retry.type = 'button';
        retry.addEventListener('click', safe(function () {
          Proctor.buildPanel();
          if (Proctor.overlay) {
            Proctor.overlay.innerHTML = '';
            Proctor.overlay.appendChild(Proctor._overlayStartCard());
          }
          Proctor.startMedia();
        }));
        card.appendChild(retry);
        return card;
      }

      // Blocking error in the panel...
      Proctor.panel.innerHTML = '';
      Proctor.panel.appendChild(errorCard());
      // ...and the overlay keeps covering the exam until access is granted.
      if (Proctor.overlay) {
        Proctor.overlay.innerHTML = '';
        var overlayCard = el('div', 'overlay-card');
        overlayCard.appendChild(errorCard());
        Proctor.overlay.appendChild(overlayCard);
      }
    },

    /* ------------------------------ audio ------------------------------ */

    startAudio: function (stream) {
      try {
        var Ctx = window.AudioContext || window.webkitAudioContext;
        if (!Ctx || !stream.getAudioTracks().length) return;

        Proctor.audioCtx = new Ctx();
        Proctor.audioSampleRate = Proctor.audioCtx.sampleRate;
        Proctor.audioSource = Proctor.audioCtx.createMediaStreamSource(stream);
        Proctor.audioProcessor = Proctor.audioCtx.createScriptProcessor(4096, 1, 1);

        Proctor.audioProcessor.onaudioprocess = function (e) {
          try {
            var data = e.inputBuffer.getChannelData(0);
            Proctor.audioChunks.push(new Float32Array(data));
            // Rolling volume level for the panel meter (subsampled RMS).
            var sum = 0;
            var n = 0;
            for (var i = 0; i < data.length; i += 16) {
              sum += data[i] * data[i];
              n++;
            }
            var rms = n ? Math.sqrt(sum / n) : 0;
            Proctor.audioLevel = Math.max(rms, Proctor.audioLevel * 0.85);
          } catch (err) { /* keep capturing */ }
        };

        // Route through a muted gain node so the processor runs without
        // feeding the microphone back to the speakers.
        Proctor.audioSilencer = Proctor.audioCtx.createGain();
        Proctor.audioSilencer.gain.value = 0;
        Proctor.audioSource.connect(Proctor.audioProcessor);
        Proctor.audioProcessor.connect(Proctor.audioSilencer);
        Proctor.audioSilencer.connect(Proctor.audioCtx.destination);

        if (Proctor.audioCtx.state === 'suspended') {
          Proctor.audioCtx.resume().catch(function () {});
        }
      } catch (err) {
        if (window.console && console.error) console.error('[Proctor] audio init', err);
      }
    },

    sendAudio: function () {
      if (Proctor.terminated || !Proctor.audioChunks.length) return;

      var chunks = Proctor.audioChunks;
      Proctor.audioChunks = [];
      var total = 0;
      chunks.forEach(function (c) { total += c.length; });
      var merged = new Float32Array(total);
      var offset = 0;
      chunks.forEach(function (c) {
        merged.set(c, offset);
        offset += c.length;
      });

      var samples = downsample(merged, Proctor.audioSampleRate, 16000);
      var blob = encodeWav(samples, 16000);
      var fd = new FormData();
      fd.append('session_id', String(Proctor.opts.sessionId));
      fd.append('audio', blob, 'chunk.wav');

      fetch(Proctor.opts.audioUrl, {
        method: 'POST',
        credentials: 'same-origin',
        keepalive: false,
        headers: { 'X-CSRFToken': getCookie('csrftoken') || '' },
        body: fd
      })
        .then(function (res) { return res.json(); })
        .then(safe(function (data) {
          Proctor.setConnected(true);
          if (!data || typeof data !== 'object') return;
          if (data.ok === false) {
            if (data.terminate) Proctor.terminate(data.error || 'Session ended.');
            return;
          }
          if (data.label === 'speech' || data.label === 'conversation') {
            Proctor.setChip('audio',
              data.label === 'conversation' ? 'Talking' : 'Speech',
              data.label === 'conversation' ? 'bad' : 'warn');
          } else if (data.label) {
            Proctor.setChip('audio', 'Quiet', 'ok');
          }
          if (data.alert) {
            Proctor.addWarning('Speech detected near the candidate', 'suspicious');
          }
        }))
        .catch(function () { Proctor.setConnected(false); });
    },

    updateAudioMeter: function () {
      if (!Proctor.ui || !Proctor.ui.audioFill) return;
      var pct = Math.min(100, Math.round(Proctor.audioLevel * 350));
      Proctor.ui.audioFill.style.width = pct + '%';
      Proctor.audioLevel *= 0.9; // decay so the meter falls when capture stops
    },

    /* ------------------------------ frames ----------------------------- */

    startLoops: function () {
      Proctor.frameTimer = setInterval(safe(Proctor.sendFrame), Proctor.opts.frameIntervalMs);
      Proctor.audioTimer = setInterval(safe(Proctor.sendAudio), Proctor.opts.audioChunkMs);
      Proctor.meterTimer = setInterval(safe(Proctor.updateAudioMeter), 250);
    },

    sendFrame: function () {
      if (Proctor.terminated || Proctor.frameInFlight) return;
      var v = Proctor.video;
      if (!v || v.readyState < 2 || !v.videoWidth) return;

      var image;
      try {
        Proctor.canvas.getContext('2d').drawImage(v, 0, 0, 640, 480);
        image = Proctor.canvas.toDataURL('image/jpeg', 0.7);
      } catch (err) {
        return; // frame not capturable this tick
      }

      Proctor.frameInFlight = true;
      fetch(Proctor.opts.frameUrl, {
        method: 'POST',
        credentials: 'same-origin',
        keepalive: false,
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': getCookie('csrftoken') || ''
        },
        body: JSON.stringify({ session_id: Proctor.opts.sessionId, image: image })
      })
        .then(function (res) { return res.json(); })
        .then(safe(function (data) {
          Proctor.frameInFlight = false;
          Proctor.setConnected(true);
          Proctor.handleFrameResponse(data);
        }))
        .catch(function () {
          Proctor.frameInFlight = false;
          Proctor.setConnected(false);
        });
    },

    handleFrameResponse: function (data) {
      if (!data || typeof data !== 'object') {
        Proctor.setConnected(false);
        return;
      }
      if (data.ok === false) {
        if (data.terminate) Proctor.terminate(data.error || 'Session is no longer active.');
        return;
      }

      var ch = data.channels || {};

      // Face presence
      var faceCount = typeof ch.face_count === 'number' ? ch.face_count : 1;
      if (faceCount === 0) {
        Proctor.setChip('face', 'Absent', 'bad');
      } else if (faceCount > 1) {
        Proctor.setChip('face', 'Multiple', 'bad');
      } else {
        Proctor.setChip('face', 'OK', 'ok');
      }

      // Identity (only trust the flag on frames where DeepFace actually ran)
      if (ch.identity_checked) {
        Proctor.identityState = !!ch.identity_verified;
      }
      if (Proctor.identityState === null) {
        Proctor.setChip('identity', 'Pending', 'warn');
      } else if (Proctor.identityState) {
        Proctor.setChip('identity', 'Verified', 'ok');
      } else {
        Proctor.setChip('identity', 'Failed', 'bad');
      }

      // Gaze
      Proctor.setChip('gaze', ch.looking_away ? 'Away' : 'OK',
        ch.looking_away ? 'warn' : 'ok');

      // Objects
      var phone = ch.phone_conf || 0;
      var book = ch.book_conf || 0;
      if (phone >= 0.5) {
        Proctor.setChip('objects', 'Phone!', 'bad');
      } else if (book >= 0.5) {
        Proctor.setChip('objects', 'Book', 'warn');
      } else {
        Proctor.setChip('objects', 'Clear', 'ok');
      }

      // Audio (server-side speech probability fused into the frame response)
      var speech = ch.audio_speech_prob || 0;
      Proctor.setChip('audio', speech >= 0.65 ? 'Speech' : 'Quiet',
        speech >= 0.65 ? 'warn' : 'ok');

      Proctor.updateRisk(data.risk_score, data.risk_level);

      if (Array.isArray(data.violations)) {
        data.violations.forEach(function (v) {
          if (v && v.message) Proctor.addWarning(v.message, v.severity || 'suspicious');
        });
      }

      if (data.terminate) {
        Proctor.terminate(data.terminate_reason || 'Terminated by the proctoring engine.');
      }
    },

    fetchStatus: function () {
      if (!Proctor.opts.statusUrl) return;
      fetch(Proctor.opts.statusUrl, { credentials: 'same-origin', keepalive: false })
        .then(function (res) { return res.json(); })
        .then(safe(function (data) {
          if (!data || typeof data !== 'object') return;
          if (data.ok === false) {
            if (data.terminate) Proctor.terminate(data.error || 'Session is no longer active.');
            return;
          }
          if (typeof data.risk_score === 'number') {
            Proctor.updateRisk(data.risk_score, data.risk_level);
          }
          if (typeof data.tab_switches === 'number') {
            Proctor.tabCount = data.tab_switches;
            Proctor.updateTabsChip();
          }
        }))
        .catch(function () { /* seed fetch is best-effort */ });
    },

    /* -------------------------- browser events ------------------------- */

    bindBrowserEvents: function () {
      if (Proctor.eventsBound) return;
      Proctor.eventsBound = true;

      Proctor.handlers = {
        vis: safe(function () {
          if (document.hidden && Proctor.active && !Proctor.terminated) {
            Proctor.lastHiddenAt = Date.now();
            // A pending blur report belongs to this same tab switch.
            if (Proctor.blurTimer) {
              clearTimeout(Proctor.blurTimer);
              Proctor.blurTimer = null;
            }
            Proctor.reportEvent('tab_switch', {});
          }
        }),
        blur: safe(function () {
          if (!Proctor.active || Proctor.terminated) return;
          if (Date.now() - Proctor.lastHiddenAt < 800) return;
          // Delay slightly: if a visibilitychange lands right after this
          // blur, it is a tab switch and the blur report is dropped.
          if (Proctor.blurTimer) clearTimeout(Proctor.blurTimer);
          Proctor.blurTimer = setTimeout(safe(function () {
            Proctor.blurTimer = null;
            if (Date.now() - Proctor.lastHiddenAt < 1000) return;
            Proctor.reportEvent('window_blur', {});
          }), 300);
        }),
        fullscreen: safe(function () {
          if (!document.fullscreenElement && Proctor.active && !Proctor.terminated) {
            Proctor.reportEvent('fullscreen_exit', {});
          }
        }),
        copy: safe(function (e) {
          e.preventDefault();
          if (Proctor.active && !Proctor.terminated) Proctor.reportEvent('copy', {});
        }),
        paste: safe(function (e) {
          e.preventDefault();
          if (Proctor.active && !Proctor.terminated) Proctor.reportEvent('paste', {});
        }),
        contextmenu: safe(function (e) {
          e.preventDefault();
          if (Proctor.active && !Proctor.terminated) Proctor.reportEvent('right_click', {});
        })
      };

      document.addEventListener('visibilitychange', Proctor.handlers.vis);
      window.addEventListener('blur', Proctor.handlers.blur);
      document.addEventListener('fullscreenchange', Proctor.handlers.fullscreen);
      document.addEventListener('copy', Proctor.handlers.copy);
      document.addEventListener('paste', Proctor.handlers.paste);
      document.addEventListener('contextmenu', Proctor.handlers.contextmenu);
    },

    unbindBrowserEvents: function () {
      if (!Proctor.eventsBound || !Proctor.handlers) return;
      Proctor.eventsBound = false;
      document.removeEventListener('visibilitychange', Proctor.handlers.vis);
      window.removeEventListener('blur', Proctor.handlers.blur);
      document.removeEventListener('fullscreenchange', Proctor.handlers.fullscreen);
      document.removeEventListener('copy', Proctor.handlers.copy);
      document.removeEventListener('paste', Proctor.handlers.paste);
      document.removeEventListener('contextmenu', Proctor.handlers.contextmenu);
      Proctor.handlers = null;
    },

    reportEvent: function (eventType, details) {
      fetch(Proctor.opts.eventUrl, {
        method: 'POST',
        credentials: 'same-origin',
        keepalive: false,
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': getCookie('csrftoken') || ''
        },
        body: JSON.stringify({
          session_id: Proctor.opts.sessionId,
          event_type: eventType,
          details: details || {}
        })
      })
        .then(function (res) { return res.json(); })
        .then(safe(function (data) {
          Proctor.setConnected(true);
          if (!data || typeof data !== 'object') return;
          if (data.ok === false) {
            if (data.terminate) Proctor.terminate(data.error || 'Session ended.');
            return;
          }
          if (typeof data.tab_switches === 'number') {
            Proctor.tabCount = data.tab_switches;
            Proctor.updateTabsChip();
          }
          if (data.message) {
            Proctor.addWarning(data.message, data.severity || 'suspicious');
          }
          if (data.terminate) {
            Proctor.terminate(data.message || 'Terminated due to repeated browser violations.');
          }
        }))
        .catch(function () { Proctor.setConnected(false); });
    },

    /* ----------------------------- panel UI ----------------------------- */

    setChip: function (key, text, state) {
      var chip = Proctor.ui && Proctor.ui.chips ? Proctor.ui.chips[key] : null;
      if (!chip) return;
      chip.value.textContent = text;
      chip.root.className = 'chip' + (state ? ' ' + state : '');
    },

    updateTabsChip: function () {
      var n = Proctor.tabCount;
      Proctor.setChip('tabs', String(n), n === 0 ? 'ok' : (n <= 2 ? 'warn' : 'bad'));
    },

    updateRisk: function (score, level) {
      if (!Proctor.ui || !Proctor.ui.riskFill) return;
      var s = typeof score === 'number' ? score : 0;
      var pct = Math.max(0, Math.min(100, Math.round(s * 100)));
      var cls = 'risk-green';
      if (level === 'violation' || (!level && s >= 0.70)) {
        cls = 'risk-red';
      } else if (level === 'suspicious' || (!level && s >= 0.45)) {
        cls = 'risk-amber';
      }
      Proctor.ui.riskFill.style.width = pct + '%';
      Proctor.ui.riskFill.className = 'risk-fill ' + cls;
      Proctor.ui.riskValue.textContent = pct + '%';
    },

    addWarning: function (message, severity) {
      var list = Proctor.ui ? Proctor.ui.warnings : null;
      if (!list) return;
      var placeholder = list.querySelector('li.empty');
      if (placeholder) placeholder.remove();

      var li = el('li', 'sev-' + (severity || 'info'));
      li.appendChild(el('span', 'w-time', timeStamp()));
      li.appendChild(el('span', 'w-msg', message));
      list.insertBefore(li, list.firstChild);    // newest first
      while (list.children.length > 6) {
        list.removeChild(list.lastChild);        // keep at most 6 entries
      }
    },

    setConnected: function (ok) {
      Proctor.connected = ok;
      if (Proctor.ui && Proctor.ui.banner) Proctor.ui.banner.hidden = ok;
      if (Proctor.ui && Proctor.ui.dot) {
        Proctor.ui.dot.className = 'proctor-dot ' + (ok ? 'on' : 'off');
      }
    },

    /* ---------------------------- termination --------------------------- */

    terminate: function (reason) {
      if (Proctor.terminated) return;
      Proctor.terminated = true;
      Proctor.stop();
      Proctor.showTerminated(reason);

      if (!Proctor.form) return;
      var fields = Proctor.form.querySelectorAll('input, select, textarea, button');
      fields.forEach(function (f) { f.disabled = true; });
      Proctor.form.classList.add('form-locked');

      setTimeout(safe(function () {
        // Re-enable fields just before the programmatic submit so the
        // student's answers are included in the POST.
        fields.forEach(function (f) { f.disabled = false; });
        var hidden = document.createElement('input');
        hidden.type = 'hidden';
        hidden.name = 'auto';
        hidden.value = 'terminated';
        Proctor.form.appendChild(hidden);
        Proctor.form.submit();
      }), 2500);
    },

    showTerminated: function (reason) {
      if (!Proctor.panel) return;
      Proctor.panel.innerHTML = '';
      var card = el('div', 'proctor-terminated');
      card.appendChild(el('h3', null, 'Exam Terminated'));
      card.appendChild(el('p', null, reason || 'The proctoring engine ended this session.'));
      card.appendChild(el('p', 'small text-mute',
        'Your answers are being submitted automatically…'));
      Proctor.panel.appendChild(card);
      Proctor.removeOverlay();
    },

    /* ------------------------------- stop ------------------------------- */

    /** Stop all monitoring: timers, listeners, audio graph and media tracks.
     *  Call before submitting the exam form. */
    stop: function () {
      Proctor.active = false;

      [Proctor.frameTimer, Proctor.audioTimer, Proctor.meterTimer].forEach(function (t) {
        if (t) clearInterval(t);
      });
      Proctor.frameTimer = Proctor.audioTimer = Proctor.meterTimer = null;
      if (Proctor.blurTimer) {
        clearTimeout(Proctor.blurTimer);
        Proctor.blurTimer = null;
      }

      Proctor.unbindBrowserEvents();

      try {
        if (Proctor.audioProcessor) {
          Proctor.audioProcessor.onaudioprocess = null;
          Proctor.audioProcessor.disconnect();
        }
        if (Proctor.audioSource) Proctor.audioSource.disconnect();
        if (Proctor.audioSilencer) Proctor.audioSilencer.disconnect();
        if (Proctor.audioCtx && Proctor.audioCtx.state !== 'closed') {
          Proctor.audioCtx.close().catch(function () {});
        }
      } catch (err) { /* audio graph already gone */ }
      Proctor.audioProcessor = Proctor.audioSource = Proctor.audioSilencer = null;
      Proctor.audioCtx = null;
      Proctor.audioChunks = [];

      if (Proctor.stream) {
        Proctor.stream.getTracks().forEach(function (track) {
          try { track.stop(); } catch (err) { /* already stopped */ }
        });
        Proctor.stream = null;
      }
    }
  };

  window.Proctor = Proctor;
})();
