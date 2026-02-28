// ── SOCKET ──────────────────────────────────────────────────────────────
const socket = io();

let consultationId = null;
let questions = [];
let questionsEnglish = [];       // English originals of questions (for backend)
let answers = [];
let currentQuestionIndex = 0;
let soapData = null;
let currentMode = 'doctor';

// ── Language & Translation State ─────────────────────────────────────────
let sessionLanguage = 'en';               // 'en' or 'ta' — set by first Whisper result
const translatedValues = {};              // { inputId: englishText } for Tamil→English

// ── Doctor Data ─────────────────────────────────────────────────────────
const DOCTORS_DATA = [
    {
        id: 1, name: 'Dr. Kavitha Subramanian', specialty: 'Dermatologist', experience: 6,
        rating: 4.8, reviews: 95, fee: 150, feeDisplay: '₹150',
        availability: 'Available Today', availabilityClass: 'available-today', availabilityOrder: 0,
        photo: 'https://randomuser.me/api/portraits/women/74.jpg',
        highlighted: true,
    },
    {
        id: 2, name: 'Dr. Anbuselvi Murugesan', specialty: 'Dermatologist', experience: 10,
        rating: 4.8, reviews: 340, fee: 250, feeDisplay: '₹250',
        availability: 'Available Tomorrow', availabilityClass: 'available-tomorrow', availabilityOrder: 2,
        photo: 'https://randomuser.me/api/portraits/women/44.jpg',
        highlighted: true,
    },
    {
        id: 3, name: 'Dr. Senthilkumar Palaniswamy', specialty: 'Cardiologist', experience: 8,
        rating: 4.7, reviews: 120, fee: 100, feeDisplay: '₹100',
        availability: 'Next slot: 3:30 PM', availabilityClass: 'available-slot', availabilityOrder: 1,
        photo: 'https://randomuser.me/api/portraits/men/32.jpg',
    },
    {
        id: 4, name: 'Dr. Meenakshi Sundaram', specialty: 'Pediatrician', experience: 5,
        rating: 4.9, reviews: 210, fee: 300, feeDisplay: '₹300',
        availability: 'Available Tomorrow', availabilityClass: 'available-tomorrow', availabilityOrder: 2,
        photo: 'https://randomuser.me/api/portraits/women/41.jpg',
    },
    {
        id: 5, name: 'Dr. Vijayalakshmi Govindasamy', specialty: 'General Physician', experience: 12,
        rating: 4.6, reviews: 510, fee: 200, feeDisplay: '₹200',
        availability: 'Available Today', availabilityClass: 'available-today', availabilityOrder: 0,
        photo: 'https://randomuser.me/api/portraits/women/68.jpg',
    },
    {
        id: 6, name: 'Dr. Thirumoorthy Ramasamy', specialty: 'Neurologist', experience: 15,
        rating: 4.9, reviews: 289, fee: 400, feeDisplay: '₹400',
        availability: 'Next Week', availabilityClass: 'available-slot', availabilityOrder: 3,
        photo: 'https://randomuser.me/api/portraits/men/55.jpg',
    },
];

// ── DOM Elements ─────────────────────────────────────────────────────────
const step1 = document.getElementById('step1');
const step2 = document.getElementById('step2');
const step3 = document.getElementById('step3');

const personaInput      = document.getElementById('persona');
const symptomsInput     = document.getElementById('symptoms');
const startBtn          = document.getElementById('startBtn');
const questionsContainer = document.getElementById('questionsContainer');
const progressText      = document.getElementById('progressText');
const doctorBtn         = document.getElementById('doctorBtn');
const patientBtn        = document.getElementById('patientBtn');

const reportedIssue  = document.getElementById('reportedIssue');
const keyFindings    = document.getElementById('keyFindings');
const soapA          = document.getElementById('soapA');
const soapP          = document.getElementById('soapP');
const redFlags       = document.getElementById('redFlags');
const confidence     = document.getElementById('confidence');

const newConsultationBtn = document.getElementById('newConsultationBtn');
const loadingOverlay   = document.getElementById('loadingOverlay');
const loadingText      = document.getElementById('loadingText');

// ── File upload elements ─────────────────────────────────────────────────
const fileInput         = document.getElementById('fileInput');
const dropzone          = document.getElementById('dropzone');
const dropzoneInner     = document.getElementById('dropzoneInner');
const dropzonePreview   = document.getElementById('dropzonePreview');
const uploadedFileName  = document.getElementById('uploadedFileName');
const removeFileBtn     = document.getElementById('removeFileBtn');

// ── UTILITIES ─────────────────────────────────────────────────────────────
function showLoading(message = 'Processing...') {
    loadingText.textContent = message;
    loadingOverlay.classList.remove('hidden');
}
function hideLoading() {
    loadingOverlay.classList.add('hidden');
}
function showStep(stepNum) {
    [step1, step2, step3].forEach(s => s.classList.remove('active'));
    if (stepNum === 1) step1.classList.add('active');
    if (stepNum === 2) step2.classList.add('active');
    if (stepNum === 3) step3.classList.add('active');
    window.scrollTo({ top: 0, behavior: 'smooth' });
}
function setButtonLoading(btn, loading) {
    btn.disabled = loading;
    btn.style.opacity = loading ? '0.6' : '1';
}


// ═════════════════════════════════════════════════════════════════════════════
// WHISPER-BASED SPEECH RECOGNITION (replaces browser webkitSpeechRecognition)
// ═════════════════════════════════════════════════════════════════════════════

let activeMediaRecorder = null;
let activeAudioChunks = [];
let activeMicBtn = null;

/**
 * Get the English value for a field — uses translated value if available,
 * otherwise the raw input value (assumed English).
 */
function getEnglishValue(inputId) {
    if (translatedValues[inputId] !== undefined && translatedValues[inputId] !== '') {
        return translatedValues[inputId];
    }
    const el = document.getElementById(inputId);
    return el ? el.value.trim() : '';
}

/**
 * Call the /translate endpoint.
 * @returns {Promise<string>} translated text, or original on failure
 */
async function translateText(text, source, target) {
    if (!text.trim() || source === target) return text;
    try {
        const resp = await fetch('/translate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, source, target }),
        });
        if (!resp.ok) {
            console.warn('Translation failed:', resp.status);
            return text;
        }
        const data = await resp.json();
        return data.translated_text || text;
    } catch (err) {
        console.warn('Translation error:', err);
        return text;
    }
}

/**
 * Detect if text contains Tamil characters (Unicode U+0B80–U+0BFF).
 * Returns true if more than 30% of alphabetic chars are Tamil.
 */
function containsTamil(text) {
    if (!text || !text.trim()) return false;
    let tamilCount = 0, alphaCount = 0;
    for (const ch of text) {
        if (ch >= '\u0B80' && ch <= '\u0BFF') { tamilCount++; alphaCount++; }
        else if (/[a-zA-Z]/.test(ch)) { alphaCount++; }
    }
    return alphaCount > 0 && (tamilCount / alphaCount) > 0.3;
}

/**
 * Check text for Tamil and switch session language if detected.
 * Also stores the English translation for the given input ID.
 */
async function detectAndSetTamil(inputId, text) {
    if (sessionLanguage === 'ta') return; // already Tamil
    if (containsTamil(text)) {
        sessionLanguage = 'ta';
        console.log('Session language set to Tamil (keyboard input detected)');
        // Translate to English for backend
        if (inputId) {
            const englishText = await translateText(text, 'ta', 'en');
            translatedValues[inputId] = englishText;
            console.log(`Translated [${inputId}]: "${englishText}"`);
        }
    }
}

function toggleMic(targetEl, micBtn) {
    // If currently recording on THIS button → stop
    if (activeMediaRecorder && activeMediaRecorder.state === 'recording' && activeMicBtn === micBtn) {
        activeMediaRecorder.stop();
        return;
    }

    // If recording on a DIFFERENT button → stop that first
    if (activeMediaRecorder && activeMediaRecorder.state === 'recording') {
        activeMediaRecorder.stop();
    }

    // Start new recording
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            // Prefer webm/opus, fall back to whatever browser supports
            let mimeType = 'audio/webm;codecs=opus';
            if (!MediaRecorder.isTypeSupported(mimeType)) {
                mimeType = 'audio/webm';
                if (!MediaRecorder.isTypeSupported(mimeType)) {
                    mimeType = 'audio/mp4';
                    if (!MediaRecorder.isTypeSupported(mimeType)) {
                        mimeType = '';  // let browser decide
                    }
                }
            }

            const options = mimeType ? { mimeType } : {};
            const recorder = new MediaRecorder(stream, options);
            activeMediaRecorder = recorder;
            activeAudioChunks = [];
            activeMicBtn = micBtn;

            micBtn.classList.add('listening');

            recorder.ondataavailable = (e) => {
                if (e.data.size > 0) activeAudioChunks.push(e.data);
            };

            recorder.onstop = async () => {
                // Stop all tracks to release mic
                stream.getTracks().forEach(t => t.stop());

                micBtn.classList.remove('listening');
                micBtn.classList.add('processing');

                const blob = new Blob(activeAudioChunks, {
                    type: recorder.mimeType || 'audio/webm'
                });
                activeMediaRecorder = null;
                activeAudioChunks = [];
                activeMicBtn = null;

                // Send to Whisper
                try {
                    const formData = new FormData();
                    formData.append('audio', blob, 'recording.webm');

                    const resp = await fetch('/transcribe', {
                        method: 'POST',
                        body: formData,
                    });

                    if (!resp.ok) {
                        const err = await resp.json().catch(() => ({}));
                        console.warn('Transcription failed:', err.error || resp.status);
                        micBtn.classList.remove('processing');
                        return;
                    }

                    const result = await resp.json();
                    const transcript = (result.text || '').trim();
                    const detectedLang = (result.language || 'en').trim();

                    if (!transcript) {
                        micBtn.classList.remove('processing');
                        return;
                    }

                    // Set session language on first Tamil detection
                    if (detectedLang === 'ta' && sessionLanguage === 'en') {
                        sessionLanguage = 'ta';
                        console.log('Session language set to Tamil');
                    }

                    // Display the transcribed text (Tamil or English as spoken)
                    if (targetEl.tagName.toLowerCase() === 'textarea') {
                        targetEl.value += (targetEl.value ? ' ' : '') + transcript;
                    } else {
                        targetEl.value = transcript;
                    }
                    targetEl.dispatchEvent(new Event('input'));

                    // If Tamil → translate to English in background and store
                    if (detectedLang === 'ta') {
                        const englishText = await translateText(transcript, 'ta', 'en');
                        const inputId = targetEl.id;
                        if (inputId) {
                            // For textarea, append to existing translation
                            if (targetEl.tagName.toLowerCase() === 'textarea' && translatedValues[inputId]) {
                                translatedValues[inputId] += ' ' + englishText;
                            } else {
                                translatedValues[inputId] = englishText;
                            }
                        }
                        console.log(`Translated [${inputId}]: "${englishText}"`);
                    } else {
                        // English speech — clear any stale translation
                        const inputId = targetEl.id;
                        if (inputId) {
                            delete translatedValues[inputId];
                        }
                    }

                } catch (err) {
                    console.error('Mic processing error:', err);
                } finally {
                    micBtn.classList.remove('processing');
                }
            };

            recorder.onerror = (e) => {
                console.warn('MediaRecorder error:', e.error);
                stream.getTracks().forEach(t => t.stop());
                micBtn.classList.remove('listening');
                activeMediaRecorder = null;
                activeMicBtn = null;
            };

            recorder.start();
        })
        .catch(err => {
            console.error('Microphone access denied:', err);
            alert('Microphone access is required for voice input. Please allow microphone access and try again.');
        });
}


// ── STEP 1: Wire up mic buttons ───────────────────────────────────────────
document.querySelectorAll('#step1 .mic-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        e.preventDefault();
        const targetId = btn.dataset.target;
        const targetEl = document.getElementById(targetId);
        if (targetEl) toggleMic(targetEl, btn);
    });
});

// ── FILE UPLOAD LOGIC ─────────────────────────────────────────────────────
function showFilePreview(fileName) {
    dropzoneInner.classList.add('hidden');
    dropzonePreview.classList.remove('hidden');
    uploadedFileName.textContent = fileName;
}

function clearFilePreview() {
    dropzoneInner.classList.remove('hidden');
    dropzonePreview.classList.add('hidden');
    uploadedFileName.textContent = '';
    fileInput.value = '';
}

fileInput.addEventListener('change', () => {
    if (fileInput.files && fileInput.files[0]) {
        showFilePreview(fileInput.files[0].name);
    }
});

removeFileBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    clearFilePreview();
});

// Drag-and-drop
dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('drag-over');
});

dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('drag-over');
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files && files[0]) {
        try {
            const dt = new DataTransfer();
            dt.items.add(files[0]);
            fileInput.files = dt.files;
        } catch (_) { /* DataTransfer not supported in all browsers */ }
        showFilePreview(files[0].name);
    }
});

// ── VIEW SWITCHER ──────────────────────────────────────────────────────────
window.switchView = function(mode) {
    currentMode = mode;
    const doctorView = document.getElementById('doctorView');
    const patientView = document.getElementById('patientView');
    const tabDoctor  = document.getElementById('tabDoctor');
    const tabPatient = document.getElementById('tabPatient');
    const doctorsSection = document.getElementById('doctorsSection');

    if (mode === 'doctor') {
        doctorView.style.display = 'block';
        patientView.style.display = 'none';
        tabDoctor.classList.add('active-tab');
        tabDoctor.style.display = '';
        tabPatient.style.display = 'none';
        document.getElementById('step3Title').textContent = 'Consultation Notes';
        if (doctorsSection) doctorsSection.style.display = 'none';
    } else {
        doctorView.style.display = 'none';
        patientView.style.display = 'block';
        tabPatient.classList.add('active-tab');
        tabPatient.style.display = '';
        tabDoctor.style.display = 'none';
        document.getElementById('step3Title').textContent = sessionLanguage === 'ta'
            ? 'நோயாளி சுருக்கம்'
            : 'Patient Summary';
        if (doctorsSection) doctorsSection.style.display = '';
    }
};

// ── POPULATE PATIENT VIEW ──────────────────────────────────────────────────
async function populatePatientView(data) {
    const fields = ['diagnosis', 'findings', 'how_found', 'treatment', 'recovery'];
    const domIds = {
        diagnosis: 'patientDiagnosis',
        findings:  'patientFindings',
        how_found: 'patientHowFound',
        treatment: 'patientTreatment',
        recovery:  'patientRecovery',
    };

    // Set English text first as immediate feedback
    for (const key of fields) {
        const el = document.getElementById(domIds[key]);
        if (el) el.innerText = data[key] && data[key].trim() !== '' ? data[key] : 'Not Available';
    }

    // If session is Tamil → translate patient summary fields to Tamil in parallel
    if (sessionLanguage === 'ta') {
        const translatePromises = fields
            .filter(k => data[k] && data[k].trim())
            .map(async (key) => {
                try {
                    const tamil = await translateText(data[key], 'en', 'ta');
                    const el = document.getElementById(domIds[key]);
                    if (el && tamil) el.innerText = tamil;
                } catch (err) {
                    console.warn(`Failed to translate ${key}:`, err);
                }
            });
        await Promise.all(translatePromises);
    }

    // Urgency badge
    const urgencyEl = document.getElementById('patientUrgency');
    if (urgencyEl) {
        const urgency = (data.urgency || '').toLowerCase();
        urgencyEl.classList.remove('urgency-low', 'urgency-medium', 'urgency-high');

        if (urgency.includes('low')) {
            urgencyEl.classList.add('urgency-low');
            urgencyEl.innerText = sessionLanguage === 'ta' ? 'குறைவு' : 'LOW';
        } else if (urgency.includes('medium') || urgency.includes('moderate')) {
            urgencyEl.classList.add('urgency-medium');
            urgencyEl.innerText = sessionLanguage === 'ta' ? 'நடுத்தரம்' : 'MEDIUM';
        } else if (urgency.includes('high') || urgency.includes('critical')) {
            urgencyEl.classList.add('urgency-high');
            urgencyEl.innerText = sessionLanguage === 'ta' ? 'அதிகம்' : 'HIGH';
        } else {
            if (sessionLanguage === 'ta' && data.urgency) {
                const tamilUrgency = await translateText(data.urgency, 'en', 'ta');
                urgencyEl.innerText = tamilUrgency || data.urgency || 'Not Specified';
            } else {
                urgencyEl.innerText = data.urgency || 'Not Specified';
            }
        }
    }
}

// ── STEP 1: START CONSULTATION ─────────────────────────────────────────────
startBtn.addEventListener('click', async () => {
    if (!personaInput.value.trim() || !symptomsInput.value.trim()) {
        alert('Please fill in both fields');
        return;
    }

    setButtonLoading(startBtn, true);
    showLoading('Starting consultation...');

    // Detect Tamil from keyboard-typed text (if not already set by mic)
    if (sessionLanguage === 'en') {
        if (containsTamil(personaInput.value)) {
            sessionLanguage = 'ta';
            console.log('Session language set to Tamil (persona keyboard input)');
            loadingText.textContent = 'தமிழ் கண்டறியப்பட்டது... மொழிபெயர்க்கிறது...';
            const engPersona = await translateText(personaInput.value.trim(), 'ta', 'en');
            translatedValues['persona'] = engPersona;
        }
        if (containsTamil(symptomsInput.value)) {
            sessionLanguage = 'ta';
            console.log('Session language set to Tamil (symptoms keyboard input)');
            loadingText.textContent = 'தமிழ் கண்டறியப்பட்டது... மொழிபெயர்க்கிறது...';
            const engSymptoms = await translateText(symptomsInput.value.trim(), 'ta', 'en');
            translatedValues['symptoms'] = engSymptoms;
        }
    }

    // Use English-translated values if user spoke/typed Tamil, otherwise raw input
    const persona  = getEnglishValue('persona') || personaInput.value.trim();
    const symptoms = getEnglishValue('symptoms') || symptomsInput.value.trim();

    try {
        const response = await fetch('/start_consultation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ persona, symptoms })
        });
        if (!response.ok) throw new Error('Failed to start consultation');
        const data = await response.json();
        consultationId = data.consultation_id;
        socket.emit('request_questions', { consultation_id: consultationId });
    } catch (error) {
        console.error(error);
        alert('Error starting consultation. Please try again.');
        hideLoading();
        setButtonLoading(startBtn, false);
    }
});

// ── SOCKET: QUESTIONS ───────────────────────────────────────────────────────
socket.on('questions_ready', async (data) => {
    questions = data.questions;
    questionsEnglish = [...data.questions];  // keep English originals for backend
    answers = [];
    currentQuestionIndex = 0;

    // If session is Tamil → translate questions to Tamil BEFORE showing Step 2
    if (sessionLanguage === 'ta') {
        loadingText.textContent = 'கேள்விகளை தமிழில் மொழிபெயர்க்கிறது...';
        const tamilQuestions = await Promise.all(
            questions.map(q => translateText(q, 'en', 'ta'))
        );
        for (let i = 0; i < questions.length; i++) {
            if (tamilQuestions[i] && tamilQuestions[i] !== questions[i]) {
                questions[i] = tamilQuestions[i];
            }
        }
    }

    hideLoading();
    setButtonLoading(startBtn, false);
    showStep(2);
    renderConversation();
});

// ── CONVERSATION ───────────────────────────────────────────────────────────
function renderConversation() {
    questionsContainer.innerHTML = '';
    for (let i = 0; i < currentQuestionIndex; i++) {
        renderQuestion(i); renderAnswer(i);
    }
    if (currentQuestionIndex < questions.length) {
        renderQuestion(currentQuestionIndex);
        updateProgress();
        doctorBtn.style.display = 'none';
        patientBtn.style.display = 'none';
    } else {
        progressText.textContent = sessionLanguage === 'ta'
            ? 'அனைத்து கேள்விகளும் பதிலளிக்கப்பட்டன! மதிப்பீடு உருவாக்கத் தயார்.'
            : 'All questions answered! Ready to generate assessment.';
        doctorBtn.style.display = 'flex';
        patientBtn.style.display = 'flex';
    }
}

function renderQuestion(index) {
    const placeholderText = sessionLanguage === 'ta'
        ? 'உங்கள் பதிலை தட்டச்சு செய்யவும் அல்லது மைக் பயன்படுத்தவும்…'
        : 'Type your answer or use the mic…';

    const questionDiv = document.createElement('div');
    questionDiv.className = 'chat-message';
    questionDiv.innerHTML = `
        <div class="chat-avatar avatar-doctor">Dr</div>
        <div class="chat-content">
            <div class="chat-bubble bubble-doctor">
                <div class="question-text">
                    <span class="question-number">${index + 1}</span>
                    ${questions[index]}
                </div>
            </div>
            ${index === currentQuestionIndex ? `
            <div class="answer-input-wrapper">
                <input type="text" class="answer-input" id="answerInput${index}"
                    placeholder="${placeholderText}" autocomplete="off">
                <button class="answer-mic-btn" id="micBtn${index}" type="button" title="Speak your answer">
                    <svg class="mic-icon" width="16" height="16" fill="currentColor" viewBox="0 0 20 20">
                        <path fill-rule="evenodd"
                            d="M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4zm-3 6a1 1 0 012 0 4 4 0 008 0 1 1 0 112 0 6 6 0 01-5 5.917V17h2a1 1 0 110 2H8a1 1 0 110-2h2v-1.083A6 6 0 014 10z"
                            clip-rule="evenodd" />
                    </svg>
                </button>
                <button class="answer-send-btn" id="sendBtn${index}" type="button" title="Send answer">
                    <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                            d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"/>
                    </svg>
                </button>
            </div>` : ''}
        </div>`;
    questionsContainer.appendChild(questionDiv);

    if (index === currentQuestionIndex) {
        const input   = document.getElementById(`answerInput${index}`);
        const micBtn  = document.getElementById(`micBtn${index}`);
        const sendBtn = document.getElementById(`sendBtn${index}`);

        const submitAnswer = async () => {
            const displayAnswer = input.value.trim();
            if (!displayAnswer) { alert('Please enter an answer'); return; }

            // Stop mic if active
            if (activeMediaRecorder && activeMediaRecorder.state === 'recording' && activeMicBtn === micBtn) {
                activeMediaRecorder.stop();
                await new Promise(r => setTimeout(r, 500));
            }

            // Detect Tamil from keyboard-typed answer
            const inputId = `answerInput${index}`;
            if (sessionLanguage === 'en' && containsTamil(displayAnswer)) {
                sessionLanguage = 'ta';
                console.log('Session language set to Tamil (answer keyboard input)');
            }

            // Get English version for the backend
            let englishAnswer = getEnglishValue(inputId);
            if (!englishAnswer || englishAnswer === displayAnswer) {
                if (sessionLanguage === 'ta') {
                    englishAnswer = await translateText(displayAnswer, 'ta', 'en');
                } else {
                    englishAnswer = displayAnswer;
                }
            }

            answers.push(displayAnswer);  // display version (Tamil or English)
            input.disabled = true; sendBtn.disabled = true; micBtn.disabled = true;

            // Send ENGLISH answer to backend
            socket.emit('submit_answer', {
                consultation_id: consultationId,
                answer: englishAnswer
            });

            delete translatedValues[inputId];
            currentQuestionIndex++;
            setTimeout(() => renderConversation(), 300);
        };

        // Mic button in Step 2
        micBtn.addEventListener('click', (e) => {
            e.preventDefault();
            toggleMic(input, micBtn);
        });

        sendBtn.addEventListener('click', submitAnswer);
        input.addEventListener('keypress', e => { if (e.key === 'Enter') submitAnswer(); });
        setTimeout(() => input.focus(), 100);
    }
}

function renderAnswer(index) {
    const answerDiv = document.createElement('div');
    answerDiv.className = 'chat-message';
    answerDiv.innerHTML = `
        <div class="chat-avatar avatar-patient">You</div>
        <div class="chat-content">
            <div class="chat-bubble bubble-patient">
                <div class="question-text">${answers[index]}</div>
            </div>
        </div>`;
    questionsContainer.appendChild(answerDiv);
}

function updateProgress() {
    const q = currentQuestionIndex + 1;
    const t = questions.length;
    progressText.textContent = sessionLanguage === 'ta'
        ? `கேள்வி ${q} / ${t}`
        : `Question ${q} of ${t}`;
}

// ── ASSESSMENT BUTTONS ─────────────────────────────────────────────────────
doctorBtn.addEventListener('click', () => {
    setButtonLoading(doctorBtn, true); setButtonLoading(patientBtn, true);
    showLoading('Generating doctor assessment...');
    currentMode = 'doctor';
    socket.emit('generate_soap', { consultation_id: consultationId });
});

patientBtn.addEventListener('click', () => {
    setButtonLoading(doctorBtn, true); setButtonLoading(patientBtn, true);
    showLoading(sessionLanguage === 'ta'
        ? 'நோயாளி சுருக்கம் உருவாக்கப்படுகிறது...'
        : 'Generating patient summary...');
    currentMode = 'patient';
    socket.emit('generate_patient_summary', { consultation_id: consultationId });
});

socket.on('soap_progress', data => { loadingText.textContent = data.message; });

socket.on('soap_generated', async (data) => {
    soapData = data;

    // Doctor view — always English
    reportedIssue.textContent = data.reported_issue;
    keyFindings.textContent   = data.key_findings;
    soapA.textContent         = data.soap.A;
    soapP.textContent         = data.soap.P;
    redFlags.textContent      = data.soap.red_flags;
    redFlags.style.color      = data.soap.red_flags === 'Yes' ? '#EF4444' : '#10B981';
    confidence.textContent    = data.soap.confidence;

    // Patient view — translate to Tamil if needed (keep loading overlay visible)
    if (sessionLanguage === 'ta' && currentMode === 'patient') {
        loadingText.textContent = 'தமிழில் மொழிபெயர்க்கிறது...';
    }
    await populatePatientView(data);

    hideLoading();
    setButtonLoading(doctorBtn, false); setButtonLoading(patientBtn, false);
    showStep(3);
    switchView(currentMode);
    renderDoctors();
});

socket.on('error', data => {
    hideLoading();
    setButtonLoading(doctorBtn, false); setButtonLoading(patientBtn, false);
    alert('Error: ' + data.message);
});

// ── NEW CONSULTATION ───────────────────────────────────────────────────────
newConsultationBtn.addEventListener('click', () => {
    consultationId = null;
    questions = []; questionsEnglish = [];
    answers = []; currentQuestionIndex = 0;
    soapData = null; currentMode = 'doctor';
    sessionLanguage = 'en';
    // Reset language toggle UI if it was set to Tamil
    if (uiLanguageActive) {
        uiLanguageActive = false;
        const langBtn = document.getElementById('langToggleBtn');
        const langLabel = document.getElementById('langLabel');
        const langText = document.getElementById('langToggleText');
        if (langBtn) langBtn.classList.remove('active');
        if (langLabel) langLabel.textContent = 'English';
        if (langText) langText.textContent = 'EN → தமிழ்';
        applyUILanguage('en');
    }
    Object.keys(translatedValues).forEach(k => delete translatedValues[k]);
    personaInput.value = ''; symptomsInput.value = '';
    clearFilePreview();
    questionsContainer.innerHTML = '';
    doctorBtn.style.display = 'none';
    patientBtn.style.display = 'none';
    showStep(1);
    personaInput.focus();
});

// ── INITIAL FOCUS ─────────────────────────────────────────────────────────
personaInput.focus();

// ── DOCTOR CARDS ──────────────────────────────────────────────────────────
window.filterAndSort = function() {
    const query   = (document.getElementById('doctorSearch')?.value || '').trim().toLowerCase();
    const sortVal = document.getElementById('doctorSort')?.value || 'recommended';

    let result = DOCTORS_DATA.filter(d => {
        if (!query) return true;
        return d.name.toLowerCase().includes(query) || d.specialty.toLowerCase().includes(query);
    });

    result = sortDoctors(result, sortVal);
    renderGrid(result);
};

function sortDoctors(list, sortVal) {
    const clone = [...list];
    switch (sortVal) {
        case 'recommended':    return clone.sort((a,b) => a.id - b.id);
        case 'rating-desc':    return clone.sort((a,b) => b.rating - a.rating || b.reviews - a.reviews);
        case 'fee-asc':        return clone.sort((a,b) => a.fee - b.fee);
        case 'fee-desc':       return clone.sort((a,b) => b.fee - a.fee);
        case 'experience-desc':return clone.sort((a,b) => b.experience - a.experience);
        case 'availability':   return clone.sort((a,b) => a.availabilityOrder - b.availabilityOrder);
        default: return clone;
    }
}

function renderGrid(doctors) {
    const grid       = document.getElementById('doctorsGrid');
    const emptyState = document.getElementById('doctorsEmpty');
    const countBadge = document.getElementById('doctorsCount');
    if (!grid) return;

    if (doctors.length === 0) {
        grid.innerHTML = ''; emptyState?.classList.remove('hidden');
        if (countBadge) countBadge.textContent = '0 doctors';
        return;
    }

    emptyState?.classList.add('hidden');
    if (countBadge) countBadge.textContent = `${doctors.length} doctor${doctors.length !== 1 ? 's' : ''}`;
    grid.innerHTML = doctors.map(d => buildCard(d)).join('');
}

function buildCard(d) {
    const isHighlighted = d.highlighted === true;
    const cardClass = isHighlighted ? 'doctor-card doctor-card-highlighted' : 'doctor-card';

    const avatarHtml = d.photo
        ? `<div class="doctor-avatar has-photo"><img src="${d.photo}" alt="${d.name}" loading="lazy"></div>`
        : `<div class="doctor-avatar">
               <svg width="36" height="36" fill="currentColor" viewBox="0 0 20 20">
                   <path fill-rule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clip-rule="evenodd"/>
               </svg>
           </div>`;

    const tagHtml = isHighlighted
        ? `<span class="highly-recommended-tag">
               <svg width="12" height="12" fill="currentColor" viewBox="0 0 20 20">
                   <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/>
               </svg>
               Highly Recommended
           </span>`
        : '';

    const feeHtml = d.feeFree
        ? `<span class="doctor-fee free">Free</span>`
        : `<span class="doctor-fee">${d.feeDisplay}</span>`;

    const starsHtml = renderStars(d.rating);

    return `
        <div class="${cardClass}" data-id="${d.id}">
            ${tagHtml}
            <div class="doctor-info">
                ${avatarHtml}
                <div class="doctor-details">
                    <h4>${d.name}</h4>
                    <p class="doctor-spec">${d.specialty} • ${d.experience} yrs</p>
                    <div class="doctor-rating">
                        ${starsHtml}
                        <span class="rating-value">${d.rating}</span>
                        <span class="review-count">(${d.reviews})</span>
                    </div>
                </div>
            </div>
            <div class="doctor-footer">
                <span class="availability ${d.availabilityClass}">${d.availability}</span>
                ${feeHtml}
            </div>
            <div class="doctor-actions">
                <button class="btn-outline-doctor" onclick="viewProfile(${d.id})">Profile</button>
                <button class="btn-book" onclick="bookAppointment('${d.name}', ${d.id})">
                    Book an Appointment
                </button>
            </div>
        </div>`;
}

function renderStars(rating) {
    let html = '';
    for (let i = 1; i <= 5; i++) {
        if (rating >= i) {
            html += `<span class="star filled">★</span>`;
        } else if (rating >= i - 0.5) {
            html += `<span class="star half">★</span>`;
        } else {
            html += `<span class="star empty">☆</span>`;
        }
    }
    return html;
}

function renderDoctors() {
    const searchEl = document.getElementById('doctorSearch');
    const sortEl   = document.getElementById('doctorSort');
    if (searchEl) searchEl.value = '';
    if (sortEl)   sortEl.value   = 'recommended';
    filterAndSort();
}

// ── Doctor Actions ────────────────────────────────────────────────────────────
window.bookAppointment = function(doctorName, doctorId) {
    const toast = document.getElementById('bookingToast');
    const msg   = document.getElementById('bookingToastMsg');
    if (!toast || !msg) return;

    const card = document.querySelector(`.doctor-card[data-id="${doctorId}"]`);
    if (card) {
        const bookBtn = card.querySelector('.btn-book');
        if (bookBtn) {
            const original = bookBtn.textContent;
            bookBtn.textContent = '✓ Requested';
            bookBtn.disabled = true;
            bookBtn.style.background = 'linear-gradient(135deg, #10B981, #059669)';
            setTimeout(() => {
                bookBtn.textContent = original;
                bookBtn.disabled = false;
                bookBtn.style.background = '';
            }, 5000);
        }
    }

    // ── Helper: send notification with given SOAP clinical data ──────
    function sendBookingNotification(clinicalData) {
        const patientInfo = personaInput ? personaInput.value.trim() : '';
        const doctor = DOCTORS_DATA.find(d => d.id === doctorId);

        fetch('/book_appointment', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                doctor_name: doctorName,
                doctor_id: doctorId,
                specialty: doctor ? doctor.specialty : '',
                fee: doctor ? doctor.feeDisplay : '',
                availability: doctor ? doctor.availability : '',
                patient_info: patientInfo,
                clinical: clinicalData,
            }),
        })
        .then(resp => resp.json())
        .then(data => {
            console.log('Booking notification triggered:', data.message || data.status);
            hideLoading();
            msg.textContent = `Appointment confirmed for ${doctorName}!`;
            toast.classList.remove('hidden');
            clearTimeout(window._toastTimer);
            window._toastTimer = setTimeout(() => toast.classList.add('hidden'), 4000);
        })
        .catch(err => {
            console.warn('Booking notification request failed:', err);
            hideLoading();
        });
    }

    // ── Check if full SOAP (A + P) is already available ─────────────
    const hasFullSoap = soapData && soapData.soap && soapData.soap.A && soapData.soap.A.trim() !== '';

    if (hasFullSoap) {
        // SOAP already generated (Doctor Assessment mode) — send directly
        const clinicalData = {
            reported_issue: soapData.reported_issue || '',
            key_findings:   soapData.key_findings || '',
            assessment:     soapData.soap.A || '',
            plan:           soapData.soap.P || '',
            red_flags:      soapData.soap.red_flags || '',
            confidence:     soapData.soap.confidence || '',
        };
        msg.textContent = `Sending appointment to ${doctorName}...`;
        toast.classList.remove('hidden');
        sendBookingNotification(clinicalData);

    } else {
        // SOAP not available (Patient Summary mode) — generate it now
        showLoading('booking an appointment...');
        msg.textContent = `Preparing clinical report for ${doctorName}...`;
        toast.classList.remove('hidden');
        clearTimeout(window._toastTimer);

        // Listen for SOAP result (one-time listener)
        socket.once('soap_for_booking_ready', function(data) {
            console.log('SOAP for booking received:', data);
            const clinicalData = {
                reported_issue: data.reported_issue || '',
                key_findings:   data.key_findings || '',
                assessment:     data.assessment || '',
                plan:           data.plan || '',
                red_flags:      data.red_flags || '',
                confidence:     data.confidence || '',
            };

            // Also update the global soapData so subsequent bookings don't re-generate
            if (soapData) {
                soapData.soap = soapData.soap || {};
                soapData.soap.A = data.assessment || '';
                soapData.soap.P = data.plan || '';
                soapData.soap.red_flags = data.red_flags || '';
                soapData.soap.confidence = data.confidence || '';
                soapData.reported_issue = data.reported_issue || soapData.reported_issue;
                soapData.key_findings = data.key_findings || soapData.key_findings;
            }

            sendBookingNotification(clinicalData);
        });

        // Trigger SOAP generation on backend
        socket.emit('generate_soap_for_booking', {
            consultation_id: consultationId,
        });
    }
};

window.viewProfile = function(doctorId) {
    const doctor = DOCTORS_DATA.find(d => d.id === doctorId);
    if (!doctor) return;
    const toast = document.getElementById('bookingToast');
    const msg   = document.getElementById('bookingToastMsg');
    if (!toast || !msg) return;
    msg.textContent = `Viewing profile for ${doctor.name} — ${doctor.specialty}, ${doctor.experience} yrs exp.`;
    toast.classList.remove('hidden');
    clearTimeout(window._toastTimer);
    window._toastTimer = setTimeout(() => toast.classList.add('hidden'), 4000);
};

window.clearDoctorSearch = function() {
    const searchEl = document.getElementById('doctorSearch');
    if (searchEl) { searchEl.value = ''; searchEl.focus(); }
    filterAndSort();
};
// ── LANGUAGE TOGGLE (UI Only — client-side dictionary) ─────────────────────
let uiLanguageActive = false; // false = English, true = Tamil UI

/**
 * Tamil translations for all static UI text.
 * Keys are CSS selectors or special identifiers; values are Tamil text.
 * This runs entirely client-side — no model or server calls needed.
 */
const UI_TRANSLATIONS = {
    // ── Step 1 ────────────────────────────────────────────────────────
    texts: {
        'OpdDoc':                       'OpdDoc',
        'AI-Powered Clinical Triage':   'AI-Powered Clinical Triage',
        'Step 1 of 3':                  'படி 1 / 3',
        'Step 2 of 3':                  'படி 2 / 3',
        'Step 3 of 3':                  'படி 3 / 3',
        'Tell us about yourself':       'உங்களைப் பற்றி சொல்லுங்கள்',
        'About You':                    'உங்களைப் பற்றி',
        'Your Symptoms':                'உங்கள் அறிகுறிகள்',
        'Start Assessment':             'மதிப்பீட்டை தொடங்கு',
        'or':                           'அல்லது',
        'Upload a Report':              'அறிக்கையை பதிவேற்று',
        'Blood test reports or X-ray images only': 'இரத்த பரிசோதனை அறிக்கைகள் அல்லது எக்ஸ்-ரே படங்கள் மட்டும்',
        'Tap to upload or drag & drop': 'பதிவேற்ற தட்டவும் அல்லது இழுத்து விடவும்',
        'JPG, PNG, PDF · Max 10 MB':    'JPG, PNG, PDF · அதிகபட்சம் 10 MB',
        // Step 1 helper texts
        'Age, gender, medical conditions, medications': 'வயது, பாலினம், மருத்துவ நிலைமைகள், மருந்துகள்',
        'Be specific about location, duration, severity': 'இடம், கால அளவு, தீவிரம் குறித்து குறிப்பிடவும்',

        // ── Step 2 ────────────────────────────────────────────────────
        'Follow-up Questions':          'தொடர் கேள்விகள்',
        'Loading...':                   'ஏற்றுகிறது...',
        'Doctor Assessment':            'மருத்துவர் மதிப்பீடு',
        'Patient Summary':              'நோயாளி சுருக்கம்',

        // ── Step 3 ────────────────────────────────────────────────────
        'Consultation Notes':           'ஆலோசனை குறிப்புகள்',
        'Reported Issue':               'தெரிவிக்கப்பட்ட பிரச்சனை',
        'Key Findings':                 'முக்கிய கண்டுபிடிப்புகள்',
        'Assessment':                   'மதிப்பீடு',
        'Plan':                         'திட்டம்',
        'Red Flags:':                   'சிவப்புக் கொடிகள்:',
        'Confidence:':                  'நம்பிக்கை:',
        'Diagnosis':                    'நோய் கண்டறிதல்',
        'Findings':                     'கண்டுபிடிப்புகள்',
        'How It Was Found':             'எவ்வாறு கண்டறியப்பட்டது',
        'Treatment Option':             'சிகிச்சை விருப்பம்',
        'Recovery Period':              'குணமாகும் காலம்',
        'Emergency Level':              'அவசர நிலை',
        'Download as PDF':              'PDF-ஆக பதிவிறக்கு',
        'New Consultation':             'புதிய ஆலோசனை',

        // ── Doctors section ───────────────────────────────────────────
        'Recommended Doctors Near You':  'உங்களுக்கு அருகிலுள்ள பரிந்துரைக்கப்பட்ட மருத்துவர்கள்',
        'No doctors match your search.': 'உங்கள் தேடலுக்கு பொருத்தமான மருத்துவர்கள் இல்லை.',
        'Clear Search':                  'தேடலை அழி',
        'Book an Appointment':           'சந்திப்பை முன்பதிவு செய்',
        'Profile':                       'சுயவிவரம்',
        'Highly Recommended':            'மிகவும் பரிந்துரைக்கப்பட்டது',

        // ── Disclaimer ────────────────────────────────────────────────
        'Medical Disclaimer:':           'மருத்துவ மறுப்பு:',

        // ── Loading overlay ───────────────────────────────────────────
        'Processing...':                 'செயலாக்கம்...',

        // ── Mode tabs ─────────────────────────────────────────────────
        '🩺 Doctor Assessment':          '🩺 மருத்துவர் மதிப்பீடு',
        '👤 Patient Summary':            '👤 நோயாளி சுருக்கம்',
    },

    // Placeholders keyed by element ID
    placeholders: {
        'persona':      'எ.கா., ஆண், 51, முன்-நீரிழிவு',
        'symptoms':     'நீங்கள் அனுபவிப்பதை விரிவாக விவரிக்கவும்...',
        'doctorSearch': 'பெயர் அல்லது நிபுணத்துவம்…',
    },

    // Select options keyed by value
    selectOptions: {
        '':             'கோப்பு வகையை தேர்வு செய்...',
        'blood':        '🩸 இரத்த பரிசோதனை அறிக்கை',
        'xray':         '🩻 எக்ஸ்-ரே படம்',
        'recommended':  'பரிந்துரைக்கப்பட்டது',
        'rating-desc':  'மதிப்பீடு: அதிகம் → குறைவு',
        'fee-asc':      'கட்டணம்: குறைவு → அதிகம்',
        'fee-desc':     'கட்டணம்: அதிகம் → குறைவு',
        'experience-desc': 'அனுபவம்: அதிகம்',
        'availability': 'கிடைக்கும் நிலை: விரைவில்',
    },
};

// Store original English values so we can restore them
const _originalTexts = new Map();
const _originalPlaceholders = new Map();
const _originalSelectOptions = new Map();

/**
 * Walk all text nodes and UI elements, applying Tamil or restoring English.
 * This is purely client-side — no server/model calls.
 */
function applyUILanguage(toLang) {
    const dict = UI_TRANSLATIONS.texts;
    const phDict = UI_TRANSLATIONS.placeholders;
    const selDict = UI_TRANSLATIONS.selectOptions;

    if (toLang === 'ta') {
        // ── Translate text nodes ──────────────────────────────────────
        const walker = document.createTreeWalker(
            document.body, NodeFilter.SHOW_TEXT, null, false
        );
        while (walker.nextNode()) {
            const node = walker.currentNode;
            const trimmed = node.textContent.trim();
            if (!trimmed) continue;
            if (dict[trimmed]) {
                if (!_originalTexts.has(node)) _originalTexts.set(node, node.textContent);
                node.textContent = node.textContent.replace(trimmed, dict[trimmed]);
            }
        }

        // ── Translate placeholders ───────────────────────────────────
        for (const [id, tamilPH] of Object.entries(phDict)) {
            const el = document.getElementById(id);
            if (el && el.placeholder) {
                if (!_originalPlaceholders.has(el)) _originalPlaceholders.set(el, el.placeholder);
                el.placeholder = tamilPH;
            }
        }

        // ── Translate select options ─────────────────────────────────
        document.querySelectorAll('select option').forEach(opt => {
            const val = opt.value;
            if (selDict[val] !== undefined) {
                if (!_originalSelectOptions.has(opt)) _originalSelectOptions.set(opt, opt.textContent);
                opt.textContent = selDict[val];
            }
        });

    } else {
        // ── Restore English ──────────────────────────────────────────
        for (const [node, original] of _originalTexts) {
            node.textContent = original;
        }
        _originalTexts.clear();

        for (const [el, original] of _originalPlaceholders) {
            el.placeholder = original;
        }
        _originalPlaceholders.clear();

        for (const [opt, original] of _originalSelectOptions) {
            opt.textContent = original;
        }
        _originalSelectOptions.clear();
    }
}

window.toggleLanguage = function() {
    uiLanguageActive = !uiLanguageActive;
    const btn = document.getElementById('langToggleBtn');
    const label = document.getElementById('langLabel');
    const text = document.getElementById('langToggleText');

    if (uiLanguageActive) {
        btn.classList.add('active');
        label.textContent = 'தமிழ்';
        text.textContent = 'Ta → EN';
        sessionLanguage = 'ta';
        applyUILanguage('ta');
    } else {
        btn.classList.remove('active');
        label.textContent = 'English';
        text.textContent = 'EN → தமிழ்';
        sessionLanguage = 'en';
        applyUILanguage('en');
    }
};

// ── COPY TO CLIPBOARD ──────────────────────────────────────────────────────
window.copySectionText = function(elementId, btn) {
    const el = document.getElementById(elementId);
    if (!el) return;
    const text = el.innerText || el.textContent || '';
    if (!text.trim()) return;

    navigator.clipboard.writeText(text).then(() => {
        const originalHTML = btn.innerHTML;
        btn.innerHTML = `<svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/>
        </svg>`;
        btn.classList.add('copied');
        setTimeout(() => {
            btn.innerHTML = originalHTML;
            btn.classList.remove('copied');
        }, 2000);
    }).catch(err => {
        // Fallback for older browsers
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); } catch(e) {}
        document.body.removeChild(ta);

        const originalHTML = btn.innerHTML;
        btn.innerHTML = `<svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/>
        </svg>`;
        btn.classList.add('copied');
        setTimeout(() => {
            btn.innerHTML = originalHTML;
            btn.classList.remove('copied');
        }, 2000);
    });
};

// ── PDF DOWNLOAD ───────────────────────────────────────────────────────────
window.downloadPDF = function() {
    // Determine which view is active
    const isDoctorView = currentMode === 'doctor';
    const title = isDoctorView ? 'Consultation Notes - Doctor Assessment' : 'Patient Summary';
    const date = new Date().toLocaleDateString('en-IN', { year:'numeric', month:'long', day:'numeric' });

    let content = '';

    if (isDoctorView) {
        const reportedIssueText = document.getElementById('reportedIssue')?.innerText || '';
        const keyFindingsText   = document.getElementById('keyFindings')?.innerText || '';
        const soapAText         = document.getElementById('soapA')?.innerText || '';
        const soapPText         = document.getElementById('soapP')?.innerText || '';
        const redFlagsText      = document.getElementById('redFlags')?.innerText || '';
        const confidenceText    = document.getElementById('confidence')?.innerText || '';

        content = `
            <h2 style="color:#6366F1; margin-bottom:0.25rem;">🩺 Doctor Assessment</h2>
            <p style="color:#64748b; font-size:0.85rem; margin-bottom:1.5rem;">Generated on ${date}</p>

            <div class="section"><strong>Reported Issue</strong><p>${reportedIssueText}</p></div>
            <div class="section"><strong>Key Findings</strong><p>${keyFindingsText}</p></div>
            <div class="section"><strong>Assessment (A)</strong><p>${soapAText}</p></div>
            <div class="section"><strong>Plan (P)</strong><p>${soapPText}</p></div>
            <div class="meta">
                <span><strong>Red Flags:</strong> ${redFlagsText}</span>
                <span style="margin-left:2rem;"><strong>Confidence:</strong> ${confidenceText}</span>
            </div>
        `;
    } else {
        const diagnosis  = document.getElementById('patientDiagnosis')?.innerText || '';
        const findings   = document.getElementById('patientFindings')?.innerText || '';
        const howFound   = document.getElementById('patientHowFound')?.innerText || '';
        const treatment  = document.getElementById('patientTreatment')?.innerText || '';
        const recovery   = document.getElementById('patientRecovery')?.innerText || '';
        const urgency    = document.getElementById('patientUrgency')?.innerText || '';

        content = `
            <h2 style="color:#6366F1; margin-bottom:0.25rem;">👤 Patient Summary</h2>
            <p style="color:#64748b; font-size:0.85rem; margin-bottom:1.5rem;">Generated on ${date}</p>

            <div class="section"><strong>Diagnosis</strong><p>${diagnosis}</p></div>
            <div class="section"><strong>Findings</strong><p>${findings}</p></div>
            <div class="section"><strong>How It Was Found</strong><p>${howFound}</p></div>
            <div class="section"><strong>Treatment Option</strong><p>${treatment}</p></div>
            <div class="section"><strong>Recovery Period</strong><p>${recovery}</p></div>
            <div class="section"><strong>Emergency Level</strong><p>${urgency}</p></div>
        `;
    }

    const html = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>${title}</title>
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; color:#1e293b; padding:2.5rem 3rem; line-height:1.7; }
        h1 { font-size:1.5rem; margin-bottom:0.5rem; color:#1e293b; }
        .section { margin-bottom:1.25rem; padding:1rem; background:#f8fafc; border-radius:8px; border-left:4px solid #6366F1; }
        .section strong { display:block; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.5px; color:#6366F1; margin-bottom:0.35rem; }
        .section p { color:#334155; font-size:0.95rem; }
        .meta { margin-top:1.25rem; padding:1rem; background:#f1f5f9; border-radius:8px; font-size:0.9rem; color:#475569; }
        .disclaimer { margin-top:2rem; padding:1rem; background:#fef3c7; border:1px solid #f59e0b; border-radius:8px; font-size:0.82rem; color:#92400e; }
        footer { margin-top:2rem; text-align:center; font-size:0.75rem; color:#94a3b8; }
    </style>
</head>
<body>
    ${content}
    <div class="disclaimer">
        <strong>⚠️ Medical Disclaimer:</strong> This is an AI-generated clinical impression for triage purposes only. It does NOT replace professional medical evaluation. If symptoms worsen or you have concerns, seek immediate medical attention.
    </div>
    <footer>Generated by OpdDoc AI Medical Assistant · ${date}</footer>
    <script>window.onload = function() { window.print(); }</script>
</body>
</html>`;

    const blob = new Blob([html], { type: 'text/html' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `opddoc-${isDoctorView ? 'doctor-assessment' : 'patient-summary'}-${Date.now()}.html`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 5000);
};