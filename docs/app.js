'use strict';
const $ = selector => document.querySelector(selector);
const attempts = { first: { prefix: 'wrong_try', name: 'First attempt', number: '01' }, second: { prefix: 'correct_try', name: 'Second attempt', number: '02' } };
const stages = [
  { heading: 'Start with the movement', suffix: '', label: 'ORIGINAL CLIP', description: 'Watch the sample, then reveal the pose overlay.', next: 'View skeleton' },
  { heading: 'Inspect the pose overlay', suffix: '_marked', label: 'PRERECORDED SKELETON', description: 'Review how the prerecorded skeleton follows the movement.', next: 'View sample feedback' },
  { heading: 'Turn movement into feedback', suffix: '_marked_colored', label: 'PRERECORDED FEEDBACK', description: 'Explore the sample’s color-coded feedback below.', next: 'Compare attempts' }
];
let attempt = 'first', stage = 0, objectUrl = null, localName = '', view = 'analysis';
const announce = text => { $('#announcement').textContent = text; };
function pauseVideos() { document.querySelectorAll('video').forEach(video => video.pause()); }
function releaseLocal() { if (objectUrl) URL.revokeObjectURL(objectUrl); objectUrl = null; localName = ''; $('#file-input').value = ''; }
function setView(next) {
  pauseVideos(); view = next;
  $('#analysis-view').hidden = next !== 'analysis'; $('#compare-view').hidden = next !== 'compare';
  document.querySelectorAll('nav [data-view]').forEach(button => { const current = button.dataset.view === next; button.classList.toggle('active', current); button.setAttribute('aria-pressed', current); });
  $('#play-both').textContent = 'Play both clips';
}
function feedbackCard(kind, title, text, dot) {
  const card = document.createElement('div'); card.className = 'feedback-card ' + kind;
  const label = document.createElement('strong'); const marker = document.createElement('span'); marker.className = 'status-dot ' + dot; marker.setAttribute('aria-hidden', 'true'); label.append(marker, title);
  const content = document.createElement('p'); content.textContent = text; card.append(label, content); return card;
}
function render() {
  const sample = attempts[attempt], step = stages[stage], video = $('#main-video');
  pauseVideos(); $('#media-error').hidden = true; $('#feedback').hidden = Boolean(objectUrl) || stage !== 2;
  $('#reference').hidden = true; $('#show-reference').setAttribute('aria-expanded', 'false');
  document.querySelectorAll('[data-attempt]').forEach(button => { const current = !objectUrl && button.dataset.attempt === attempt; button.classList.toggle('selected', current); button.setAttribute('aria-pressed', current); });
  document.querySelectorAll('[data-stage]').forEach(button => { const current = Number(button.dataset.stage) === stage; button.classList.toggle('current', current); button.setAttribute('aria-pressed', current); button.disabled = Boolean(objectUrl); });
  $('#clip-label').textContent = objectUrl ? 'LOCAL VIDEO / PREVIEW ONLY' : `SAMPLE ${sample.number} / PUSH-UP`;
  $('#stage-heading').textContent = objectUrl ? 'Your video, on your device' : step.heading;
  $('#step-count').textContent = objectUrl ? 'PREVIEW' : `0${stage + 1} / 03`;
  $('#video-label').textContent = objectUrl ? 'LOCAL PREVIEW · NO ANALYSIS' : step.label;
  $('#stage-description').textContent = objectUrl ? `${localName} · This file stays in your browser. Select a sample to explore analysis.` : step.description;
  $('#next-step').textContent = objectUrl ? 'Return to sample →' : step.next + ' →';
  video.poster = objectUrl ? '' : `assets/${sample.prefix}${step.suffix}.jpg`;
  video.src = objectUrl || `assets/${sample.prefix}${step.suffix}.mp4`;
  video.setAttribute('aria-label', objectUrl ? 'Your local video preview' : `${sample.name}: ${step.label.toLowerCase()}`);
  video.load();
  const cards = [feedbackCard('good', 'Good form', attempt === 'first' ? 'Triceps, core, chest' : 'Triceps, core, chest, shoulders', 'green'), feedbackCard('work', 'Needs work', 'Biceps', 'orange')];
  if (attempt === 'first') cards.push(feedbackCard('attention', 'Needs attention', 'Shoulders', 'red'));
  $('#feedback-grid').replaceChildren(...cards);
}
document.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click', () => { setView(button.dataset.view); announce(view === 'compare' ? 'Comparing the two sample attempts.' : 'Analysis view.'); }));
document.querySelectorAll('[data-attempt]').forEach(button => button.addEventListener('click', () => { releaseLocal(); attempt = button.dataset.attempt; render(); announce(`${attempts[attempt].name} selected.`); }));
document.querySelectorAll('[data-stage]').forEach(button => button.addEventListener('click', () => { stage = Number(button.dataset.stage); render(); announce(stages[stage].heading); }));
$('#next-step').addEventListener('click', () => {
  if (objectUrl) { releaseLocal(); stage = 0; render(); }
  else if (stage < 2) { stage++; render(); }
  else setView('compare');
  const heading = $(view === 'compare' ? '#compare-heading' : '#stage-heading'); heading.focus({ preventScroll: true }); announce(heading.textContent);
});
$('.reset').addEventListener('click', () => { releaseLocal(); attempt = 'first'; stage = 0; setView('analysis'); render(); announce('Demo reset to the first attempt.'); });
$('#show-reference').addEventListener('click', () => { const panel = $('#reference'); panel.hidden = !panel.hidden; $('#show-reference').setAttribute('aria-expanded', String(!panel.hidden)); if (panel.hidden) $('#reference-video').pause(); });
$('#pick-file').addEventListener('click', () => $('#file-input').click());
$('#file-input').addEventListener('change', () => {
  const file = $('#file-input').files[0]; if (!file) return;
  if (!file.type.startsWith('video/')) { $('#media-error').textContent = 'Choose a video file, such as MP4 or WebM.'; $('#media-error').hidden = false; return; }
  releaseLocal(); localName = file.name; objectUrl = URL.createObjectURL(file); stage = 0; render(); announce('Local preview loaded. No analysis is performed on this file.');
});
$('#main-video').addEventListener('error', () => { $('#media-error').textContent = objectUrl ? 'This browser cannot play that video format. Try an MP4 (H.264) or WebM file.' : 'The sample could not load. Check your connection and select the clip again.'; $('#media-error').hidden = false; });
$('#play-both').addEventListener('click', async () => {
  const videos = [$('#compare-first'), $('#compare-second')];
  if (videos.some(video => !video.paused)) { videos.forEach(video => video.pause()); $('#play-both').textContent = 'Play both clips'; return; }
  videos.forEach(video => { video.currentTime = 0; });
  const results = await Promise.allSettled(videos.map(video => video.play()));
  if (results.some(result => result.status === 'rejected')) { videos.forEach(video => video.pause()); announce('Use the play controls beneath each clip.'); $('#play-both').textContent = 'Play both clips'; }
  else $('#play-both').textContent = 'Pause both clips';
});
window.addEventListener('pagehide', () => { pauseVideos(); if (objectUrl) URL.revokeObjectURL(objectUrl); });
render();
