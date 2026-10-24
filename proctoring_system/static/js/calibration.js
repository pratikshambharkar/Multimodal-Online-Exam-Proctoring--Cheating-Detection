/**
 * RAIM — gaze calibration client (vanilla ES6, no dependencies).
 *
 * Usage (from calibration.html):
 *   Calibration.init({
 *     sessionId: 12,
 *     sampleUrl: '/proctoring/api/calibration/sample/',
 *     finishUrl: '/proctoring/api/calibration/finish/',
 *     nextUrl: '/exams/session/12/'
 *   });
 *
 * Sweeps a marker dot to ~9 on-screen positions (corners, edge midpoints,
 * center). At each, after a short settle pause, it captures a few webcam
 * frames and posts them to the server, which extracts raw gaze features
 * (yaw/pitch/iris position) via GazeTracker. Once every point has been
 * visited (or the candidate chooses to skip), it asks the server to reduce
 * the staged samples into a personalized gaze envelope and redirects into
 * the exam room. A calibration that fails or is skipped is not fatal — the
 * server falls back to its global thresholds — so this script always
 * reaches ``nextUrl`` in the end, even on camera failure.
 */
(function () {
  'use strict';

  // The top-center and bottom-center columns are nudged in from the true
  // edge (0.06/0.94) so the marker never overlaps the fixed progress bar
  // (top) or instruction text (bottom) — both are horizontally centered at
  // the same x as this column. Corner/side markers keep the tighter offset
  // since nothing else fixed sits near them.
  var POINTS = [
    { x: 0.04, y: 0.06 }, { x: 0.50, y: 0.15 }, { x: 0.96, y: 0.06 },
    { x: 0.04, y: 0.50 }, { x: 0.50, y: 0.50 }, { x: 0.96, y: 0.50 },
    { x: 0.04, y: 0.94 }, { x: 0.50, y: 0.85 }, { x: 0.96, y: 0.94 }
  ];
  var SAMPLES_PER_POINT = 5;
  var SAMPLE_INTERVAL_MS = 220;
  var SETTLE_MS = 900; // time to notice + look at the marker before sampling begins

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

  function safe(fn) {
    return function () {
      try {
        return fn.apply(this, arguments);
      } catch (err) {
        if (window.console && console.error) console.error('[Calibration]', err);
      }
    };
  }

  function postJson(url, body) {
    return fetch(url, {
      method: 'POST',
      credentials: 'same-origin',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': getCookie('csrftoken') || ''
      },
      body: JSON.stringify(body)
    }).then(function (res) { return res.json(); });
  }

  var Calibration = {
    opts: null,
    video: null,
    canvas: null,
    stream: null,
    pointIndex: 0,
    finished: false,

    intro: null,
    stage: null,
    errorCard: null,
    dot: null,
    instruction: null,
    progressFill: null,
    progressText: null,

    init: function (opts) {
      Calibration.opts = opts || {};
      Calibration.intro = document.getElementById('calib-intro');
      Calibration.stage = document.getElementById('calib-stage');
      Calibration.errorCard = document.getElementById('calib-error');
      Calibration.dot = document.getElementById('calib-dot');
      Calibration.instruction = document.getElementById('calib-instruction');
      Calibration.progressFill = document.getElementById('calib-progress-fill');
      Calibration.progressText = document.getElementById('calib-progress-text');
      Calibration.video = document.getElementById('calib-video');

      Calibration.canvas = document.createElement('canvas');
      Calibration.canvas.width = 640;
      Calibration.canvas.height = 480;

      Calibration.startBtn = document.getElementById('calib-start-btn');
      Calibration.startHint = document.getElementById('calib-start-hint');
      if (Calibration.startBtn) {
        Calibration.startBtn.addEventListener('click', safe(function () {
          // Immediate feedback: getUserMedia() only resolves once the
          // browser's own permission prompt is answered, which can take a
          // moment (or be easy to miss) - without this the button appeared
          // to do nothing at all while a request was actually pending.
          Calibration.startBtn.disabled = true;
          Calibration.startBtn.textContent = 'Requesting camera access…';
          if (Calibration.startHint) Calibration.startHint.hidden = true;
          Calibration.pendingTimer = setTimeout(safe(function () {
            if (Calibration.startHint) {
              Calibration.startHint.hidden = false;
              Calibration.startHint.textContent =
                'Still waiting — look for a camera-permission prompt near your ' +
                'browser’s address bar and click Allow.';
            }
            Calibration.pendingTimer = setTimeout(safe(function () {
              if (Calibration.startHint) {
                Calibration.startHint.textContent =
                  'Still nothing? Your camera may be blocked at the OS level, in ' +
                  'use by another app, or missing. You can use the "Skip ' +
                  'calibration" link below once the stage appears, or retry.';
              }
            }), 6000);
          }), 4000);
          Calibration.start();
        }));
      }

      ['calib-skip', 'calib-skip-intro'].forEach(function (id) {
        var link = document.getElementById(id);
        if (link) {
          link.addEventListener('click', safe(function (e) {
            e.preventDefault();
            Calibration.clearPendingTimer();
            Calibration.finish();
          }));
        }
      });
      var errorSkipBtn = document.getElementById('calib-error-skip');
      if (errorSkipBtn) errorSkipBtn.addEventListener('click', safe(Calibration.finish));
    },

    start: function () {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        Calibration.clearPendingTimer();
        Calibration.showError('This browser cannot access the camera. Please use an up-to-date Chrome or Edge.');
        return;
      }
      navigator.mediaDevices
        .getUserMedia({ video: { width: 640, height: 480 }, audio: false })
        .then(safe(function (stream) {
          Calibration.clearPendingTimer();
          Calibration.stream = stream;
          Calibration.video.srcObject = stream;
          var playPromise = Calibration.video.play();
          if (playPromise && playPromise.catch) playPromise.catch(function () {});
          if (Calibration.intro) Calibration.intro.hidden = true;
          if (Calibration.stage) Calibration.stage.hidden = false;
          Calibration.pointIndex = 0;
          Calibration.runPoint();
        }))
        .catch(safe(function () {
          Calibration.clearPendingTimer();
          Calibration.resetStartButton();
          Calibration.showError(
            'Camera access was blocked. Calibration needs the webcam to see ' +
            'where you are looking; you may continue without it.');
        }));
    },

    clearPendingTimer: function () {
      if (Calibration.pendingTimer) {
        clearTimeout(Calibration.pendingTimer);
        Calibration.pendingTimer = null;
      }
      if (Calibration.startHint) Calibration.startHint.hidden = true;
    },

    resetStartButton: function () {
      if (Calibration.startBtn) {
        Calibration.startBtn.disabled = false;
        Calibration.startBtn.textContent = 'Begin calibration';
      }
    },

    showError: function (message) {
      if (Calibration.intro) Calibration.intro.hidden = true;
      if (Calibration.stage) Calibration.stage.hidden = true;
      if (Calibration.errorCard) {
        Calibration.errorCard.hidden = false;
        var msgEl = document.getElementById('calib-error-message');
        if (msgEl) msgEl.textContent = message;
      }
    },

    runPoint: function () {
      if (Calibration.pointIndex >= POINTS.length) {
        Calibration.finish();
        return;
      }
      var point = POINTS[Calibration.pointIndex];
      if (Calibration.dot) {
        Calibration.dot.style.left = (point.x * 100) + '%';
        Calibration.dot.style.top = (point.y * 100) + '%';
      }
      if (Calibration.progressText) {
        Calibration.progressText.textContent =
          'Point ' + (Calibration.pointIndex + 1) + ' of ' + POINTS.length;
      }
      if (Calibration.progressFill) {
        var pct = Math.round((Calibration.pointIndex / POINTS.length) * 100);
        Calibration.progressFill.style.width = pct + '%';
      }
      if (Calibration.instruction) Calibration.instruction.textContent = 'Look at the dot';

      // Give the candidate a moment to move their eyes before sampling.
      setTimeout(safe(function () {
        Calibration.captureSamples(SAMPLES_PER_POINT);
      }), SETTLE_MS);
    },

    captureSamples: function (remaining) {
      if (remaining <= 0) {
        Calibration.pointIndex += 1;
        Calibration.runPoint();
        return;
      }
      var v = Calibration.video;
      var image = null;
      if (v && v.readyState >= 2 && v.videoWidth) {
        try {
          Calibration.canvas.getContext('2d').drawImage(v, 0, 0, 640, 480);
          image = Calibration.canvas.toDataURL('image/jpeg', 0.7);
        } catch (err) { image = null; }
      }

      var advance = function () {
        setTimeout(safe(function () {
          Calibration.captureSamples(remaining - 1);
        }), SAMPLE_INTERVAL_MS);
      };

      if (!image) {
        advance();
        return;
      }

      postJson(Calibration.opts.sampleUrl, {
        session_id: Calibration.opts.sessionId,
        point_index: Calibration.pointIndex,
        image: image
      }).then(advance).catch(advance);
    },

    finish: function () {
      if (Calibration.finished) return;
      Calibration.finished = true;

      Calibration.stopCamera();

      postJson(Calibration.opts.finishUrl, { session_id: Calibration.opts.sessionId })
        .then(safe(Calibration.goToExam))
        .catch(safe(Calibration.goToExam));
    },

    goToExam: function () {
      window.location.href = Calibration.opts.nextUrl;
    },

    stopCamera: function () {
      if (Calibration.stream) {
        Calibration.stream.getTracks().forEach(function (track) {
          try { track.stop(); } catch (err) { /* already stopped */ }
        });
        Calibration.stream = null;
      }
    }
  };

  window.Calibration = Calibration;
})();
