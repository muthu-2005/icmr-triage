// ── SOCKET ──────────────────────────────────────────────────────────────
const socket = io();

let consultationId = null;
let questions = [];
let answers = [];
let currentQuestionIndex = 0;
let soapData = null;
let currentMode = 'doctor';

// ── Doctor Data ─────────────────────────────────────────────────────────
const DOCTORS_DATA = [
    {
        id: 1, name: 'Dr. John Smith', specialty: 'Dermatologist', experience: 6,
        rating: 4.8, reviews: 95, fee: 150, feeDisplay: '₹150',
        availability: 'Available Today', availabilityClass: 'available-today', availabilityOrder: 0,
        photo: 'https://randomuser.me/api/portraits/women/74.jpg',
    },
    {
        id: 2, name: 'Dr. Ananya Iyer', specialty: 'Dermatologist', experience: 10,
        rating: 4.8, reviews: 340, fee: 250, feeDisplay: '₹250',
        availability: 'Available Tomorrow', availabilityClass: 'available-tomorrow', availabilityOrder: 2,
        photo: 'https://randomuser.me/api/portraits/women/44.jpg',
    },
    {
        id: 3, name: 'Dr. Sarah Khan', specialty: 'Cardiologist', experience: 8,
        rating: 4.7, reviews: 120, fee: 100, feeDisplay: '₹100',
        availability: 'Next slot: 3:30 PM', availabilityClass: 'available-slot', availabilityOrder: 1,
        photo: 'https://randomuser.me/api/portraits/women/4.jpg',
    },
    {
        id: 4, name: 'Dr. Alex Brown', specialty: 'Pediatrician', experience: 5,
        rating: 4.9, reviews: 210, fee: 300, feeDisplay: '₹300',
        availability: 'Available Tomorrow', availabilityClass: 'available-tomorrow', availabilityOrder: 2,
        photo: 'https://randomuser.me/api/portraits/women/41.jpg',
    },
    {
        id: 5, name: 'Dr. Priya Nair', specialty: 'General Physician', experience: 12,
        rating: 4.6, reviews: 510, fee: 200, feeDisplay: '₹200',
        availability: 'Available Today', availabilityClass: 'available-today', availabilityOrder: 0,
        photo: 'https://randomuser.me/api/portraits/women/68.jpg',
    },
    {
        id: 6, name: 'Dr. Rajan Mehta', specialty: 'Neurologist', experience: 15,
        rating: 4.9, reviews: 289, fee: 400, feeDisplay: '₹400',
        availability: 'Next Week', availabilityClass: 'available-slot', availabilityOrder: 3,
        photo: 'https://randomuser.me/api/portraits/women/49.jpg',
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

// ── SPEECH RECOGNITION HELPER ──────────────────────────────────────────────
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let activeRecognition = null; // track current recognition session

/**
 * Toggle speech recognition for a given target input/textarea.
 * @param {HTMLElement} targetEl  - the input or textarea to fill
 * @param {HTMLElement} micBtn    - the mic button that was clicked
 */
function toggleMic(targetEl, micBtn) {
    // If a recognition session is running for THIS button, stop it
    if (activeRecognition && micBtn.classList.contains('listening')) {
        activeRecognition.stop();
        return;
    }

    // Stop any other running session first
    if (activeRecognition) {
        activeRecognition.stop();
    }

    if (!SpeechRecognition) {
        alert('Speech recognition is not supported in your browser. Please use Chrome or Edge.');
        return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = 'en-IN';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    activeRecognition = recognition;

    micBtn.classList.add('listening');

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        if (targetEl.tagName.toLowerCase() === 'textarea') {
            targetEl.value += (targetEl.value ? ' ' : '') + transcript;
        } else {
            targetEl.value = transcript;
        }
        targetEl.dispatchEvent(new Event('input'));
    };

    recognition.onerror = (event) => {
        console.warn('Speech recognition error:', event.error);
        micBtn.classList.remove('listening');
        activeRecognition = null;
    };

    recognition.onend = () => {
        micBtn.classList.remove('listening');
        activeRecognition = null;
    };

    recognition.start();
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
        // Assign to file input (where possible)
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

    if (mode === 'doctor') {
        doctorView.style.display = 'block';
        patientView.style.display = 'none';
        tabDoctor.classList.add('active-tab');
        tabPatient.classList.remove('active-tab');
        document.getElementById('step3Title').textContent = 'Consultation Notes';
    } else {
        doctorView.style.display = 'none';
        patientView.style.display = 'block';
        tabDoctor.classList.remove('active-tab');
        tabPatient.classList.add('active-tab');
        document.getElementById('step3Title').textContent = 'Patient Summary';
    }
};

// ── POPULATE PATIENT VIEW ──────────────────────────────────────────────────
function populatePatientView(data) {
    const setText = (id, value) => {
        const el = document.getElementById(id);
        if (el) el.innerText = value && value.trim() !== '' ? value : 'Not Available';
    };

    setText('patientDiagnosis', data.diagnosis);
    setText('patientFindings',  data.findings);
    setText('patientHowFound',  data.how_found);
    setText('patientTreatment', data.treatment);
    setText('patientRecovery',  data.recovery);

    const urgencyEl = document.getElementById('patientUrgency');
    if (urgencyEl) {
        const urgency = (data.urgency || '').toLowerCase();
        urgencyEl.classList.remove('urgency-low', 'urgency-medium', 'urgency-high');

        if (urgency.includes('low')) {
            urgencyEl.classList.add('urgency-low'); urgencyEl.innerText = 'LOW';
        } else if (urgency.includes('medium') || urgency.includes('moderate')) {
            urgencyEl.classList.add('urgency-medium'); urgencyEl.innerText = 'MEDIUM';
        } else if (urgency.includes('high') || urgency.includes('critical')) {
            urgencyEl.classList.add('urgency-high'); urgencyEl.innerText = 'HIGH';
        } else {
            urgencyEl.innerText = data.urgency || 'Not Specified';
        }
    }
}

// ── STEP 1: START CONSULTATION ─────────────────────────────────────────────
startBtn.addEventListener('click', async () => {
    const persona  = personaInput.value.trim();
    const symptoms = symptomsInput.value.trim();
    if (!persona || !symptoms) { alert('Please fill in both fields'); return; }

    setButtonLoading(startBtn, true);
    showLoading('Starting consultation...');

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
socket.on('questions_ready', (data) => {
    questions = data.questions;
    answers = [];
    currentQuestionIndex = 0;
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
        progressText.textContent = 'All questions answered! Ready to generate assessment.';
        doctorBtn.style.display = 'flex';
        patientBtn.style.display = 'flex';
    }
}

function renderQuestion(index) {
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
                    placeholder="Type your answer or use the mic…" autocomplete="off">
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

        const submitAnswer = () => {
            const answer = input.value.trim();
            if (!answer) { alert('Please enter an answer'); return; }
            // Stop mic if active
            if (activeRecognition && micBtn.classList.contains('listening')) {
                activeRecognition.stop();
            }
            answers.push(answer);
            input.disabled = true; sendBtn.disabled = true; micBtn.disabled = true;
            socket.emit('submit_answer', { consultation_id: consultationId, answer });
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
    progressText.textContent = `Question ${currentQuestionIndex + 1} of ${questions.length}`;
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
    showLoading('Generating patient summary...');
    currentMode = 'patient';
    socket.emit('generate_patient_summary', { consultation_id: consultationId });
});

socket.on('soap_progress', data => { loadingText.textContent = data.message; });

socket.on('soap_generated', data => {
    hideLoading();
    setButtonLoading(doctorBtn, false); setButtonLoading(patientBtn, false);
    soapData = data;

    reportedIssue.textContent = data.reported_issue;
    keyFindings.textContent   = data.key_findings;
    soapA.textContent         = data.soap.A;
    soapP.textContent         = data.soap.P;
    redFlags.textContent      = data.soap.red_flags;
    redFlags.style.color      = data.soap.red_flags === 'Yes' ? '#EF4444' : '#10B981';
    confidence.textContent    = data.soap.confidence;

    populatePatientView(data);

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
    questions = []; answers = []; currentQuestionIndex = 0;
    soapData = null; currentMode = 'doctor';
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
        case 'rating-asc':     return clone.sort((a,b) => a.rating - b.rating || a.reviews - b.reviews);
        case 'fee-asc':        return clone.sort((a,b) => a.fee - b.fee);
        case 'fee-desc':       return clone.sort((a,b) => b.fee - a.fee);
        case 'experience-desc':return clone.sort((a,b) => b.experience - a.experience);
        case 'experience-asc': return clone.sort((a,b) => a.experience - b.experience);
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
    const avatarHtml = d.photo
        ? `<div class="doctor-avatar has-photo"><img src="${d.photo}" alt="${d.name}" loading="lazy"></div>`
        : `<div class="doctor-avatar">
               <svg width="36" height="36" fill="currentColor" viewBox="0 0 20 20">
                   <path fill-rule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clip-rule="evenodd"/>
               </svg>
           </div>`;

    const feeHtml = d.feeFree
        ? `<span class="doctor-fee free">Free</span>`
        : `<span class="doctor-fee">${d.feeDisplay}</span>`;

    const starsHtml = renderStars(d.rating);

    return `
        <div class="doctor-card" data-id="${d.id}">
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

    msg.textContent = `Appointment request sent to ${doctorName}!`;
    toast.classList.remove('hidden');
    clearTimeout(window._toastTimer);
    window._toastTimer = setTimeout(() => toast.classList.add('hidden'), 4000);
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