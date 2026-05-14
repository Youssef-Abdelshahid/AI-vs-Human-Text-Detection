const form = document.getElementById('analysisForm');
const inputText = document.getElementById('inputText');
const essayModelSelect = document.getElementById('essayModelSelect');
const sentenceModelSelect = document.getElementById('sentenceModelSelect');
const helperText = document.getElementById('helperText');
const analyzeButton = document.getElementById('analyzeButton');
const errorBox = document.getElementById('errorBox');
const emptyState = document.getElementById('emptyState');
const resultContent = document.getElementById('resultContent');
const aiPercentage = document.getElementById('aiPercentage');
const essayModelNote = document.getElementById('essayModelNote');
const meterFill = document.getElementById('meterFill');
const overallLabel = document.getElementById('overallLabel');
const humanPercentageBadge = document.getElementById('humanPercentageBadge');
const analysisNote = document.getElementById('analysisNote');
const breakdownTitle = document.getElementById('breakdownTitle');
const sentenceModelNote = document.getElementById('sentenceModelNote');
const highlightedText = document.getElementById('highlightedText');
const segments = document.getElementById('segments');
const modelBadge = document.getElementById('modelBadge');
const modelNames = new Map();
let modelsReady = false;

inputText.placeholder = 'Paste English text to analyze. The essay model gives the overall AI score; the sentence model highlights the most AI-like local sections.';
helperText.textContent = 'Overall percentage comes from the essay model. Highlighting is guided by local sentence-model ranking.';
modelBadge.textContent = 'essay + sentence models';

function showError(message) {
  errorBox.textContent = message;
  errorBox.style.display = 'block';
}

function clearError() {
  errorBox.textContent = '';
  errorBox.style.display = 'none';
}

function updateAnalyzeState(isBusy = false) {
  analyzeButton.disabled = isBusy || !modelsReady || !essayModelSelect.value || !sentenceModelSelect.value;
  if (!isBusy) {
    analyzeButton.textContent = 'Analyze';
  }
}

async function readJsonResponse(response) {
  const text = await response.text();
  if (!text) {
    return {};
  }

  try {
    return JSON.parse(text);
  } catch (error) {
    throw new Error('The server returned an invalid response. Check the Flask terminal for details.');
  }
}

function renderSegments(items, unitLabel = 'Section') {
  segments.innerHTML = '';
  items.forEach((item) => {
    const segment = document.createElement('article');
    segment.className = `segment ${item.selected ? 'ai' : 'human'}`;

    const meta = document.createElement('div');
    meta.className = 'segment-meta';

    const label = document.createElement('span');
    const selectedText = item.selected ? 'Highlighted' : 'Not highlighted';
    label.textContent = `${unitLabel} #${item.index} ${selectedText}`;

    const confidence = document.createElement('span');
    confidence.textContent = item.ai_confidence === null ? '' : `Local AI-likeness ${item.ai_confidence}%`;

    const text = document.createElement('div');
    text.textContent = item.text;

    meta.append(label, confidence);
    segment.append(meta, text);
    segments.appendChild(segment);
  });
}

function populateSelect(select, models, defaultId, emptyText) {
  select.innerHTML = '';
  if (!models.length) {
    const option = document.createElement('option');
    option.value = '';
    option.textContent = emptyText;
    select.appendChild(option);
    select.disabled = true;
    return;
  }

  models.forEach((model) => {
    modelNames.set(model.id, model.name);
    const option = document.createElement('option');
    option.value = model.id;
    option.textContent = model.name;
    select.appendChild(option);
  });

  if (defaultId && models.some((model) => model.id === defaultId)) {
    select.value = defaultId;
  }
  select.disabled = false;
}

async function loadModels() {
  clearError();
  try {
    const response = await fetch('/models');
    const data = await readJsonResponse(response);
    if (!response.ok) {
      throw new Error(data.error || 'Could not load available models.');
    }

    const essayModels = data.essay_models || [];
    const sentenceModels = data.sentence_models || [];
    populateSelect(essayModelSelect, essayModels, data.defaults?.essay_model_id, 'No essay models found');
    populateSelect(sentenceModelSelect, sentenceModels, data.defaults?.sentence_model_id, 'No sentence models found');

    if (!essayModels.length) {
      showError('No supported essay models were found in models/essay.');
    } else if (!sentenceModels.length) {
      showError('No supported sentence models were found in models/sentence.');
    }

    modelsReady = Boolean(essayModels.length && sentenceModels.length);
    updateAnalyzeState();
  } catch (error) {
    modelsReady = false;
    essayModelSelect.innerHTML = '<option value="">Essay models unavailable</option>';
    sentenceModelSelect.innerHTML = '<option value="">Sentence models unavailable</option>';
    essayModelSelect.disabled = true;
    sentenceModelSelect.disabled = true;
    updateAnalyzeState();
    showError(error.message || 'Model list loading failed. Check the Flask terminal for details.');
  }
}

essayModelSelect.addEventListener('change', () => updateAnalyzeState());
sentenceModelSelect.addEventListener('change', () => updateAnalyzeState());

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  clearError();

  const text = inputText.value.trim();
  if (!text) {
    showError('Please paste some text to analyze.');
    return;
  }
  if (!essayModelSelect.value || !sentenceModelSelect.value) {
    showError('Please choose both an essay model and a sentence model.');
    return;
  }

  analyzeButton.textContent = 'Analyzing...';
  updateAnalyzeState(true);
  const controller = new AbortController();
  const timeoutMs = 120000;
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        essay_model_id: essayModelSelect.value,
        sentence_model_id: sentenceModelSelect.value
      }),
      signal: controller.signal
    });
    const data = await readJsonResponse(response);
    if (!response.ok) {
      throw new Error(data.error || 'Analysis failed.');
    }

    emptyState.hidden = true;
    resultContent.hidden = false;
    aiPercentage.textContent = data.ai_percentage;
    essayModelNote.textContent = `Based on selected essay model: ${data.essay_model_name}`;
    essayModelNote.style.display = 'block';
    meterFill.style.width = `${Math.max(0, Math.min(100, data.ai_percentage))}%`;
    overallLabel.textContent = data.overall_label;
    if (data.human_percentage === null || data.human_percentage === undefined) {
      humanPercentageBadge.hidden = true;
    } else {
      humanPercentageBadge.hidden = false;
      humanPercentageBadge.textContent = `Human ${data.human_percentage}%`;
    }
    analysisNote.textContent = data.analysis_note || '';
    analysisNote.style.display = data.analysis_note ? 'block' : 'none';
    breakdownTitle.textContent = data.breakdown_label || 'Chunk Breakdown';
    sentenceModelNote.textContent = `Based on selected sentence model: ${data.sentence_model_name}`;
    sentenceModelNote.style.display = 'block';
    highlightedText.innerHTML = data.highlighted_text;
    modelBadge.textContent = data.essay_model_name || modelNames.get(essayModelSelect.value) || 'Selected essay model';
    helperText.textContent = `Overall score: ${data.essay_model_name}. Highlighting: ${data.sentence_model_name}.`;
    const unitLabel = (data.breakdown_label || '').startsWith('Paragraph')
      ? 'Paragraph'
      : (data.breakdown_label || '').startsWith('Document') ? 'Document' : 'Section';
    renderSegments(data.segments, unitLabel);
  } catch (error) {
    if (error.name === 'AbortError') {
      showError('Analysis is taking too long. The model may still be loading in Flask; check the terminal and try again after it finishes.');
    } else if (error instanceof TypeError) {
      showError('Could not reach the Flask server. Check the terminal to see whether the app crashed, restarted, or is still loading a large model.');
    } else {
      showError(error.message);
    }
  } finally {
    window.clearTimeout(timeout);
    updateAnalyzeState();
  }
});

loadModels();
