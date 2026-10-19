// MTranServer Web UI — main application logic (ES6 modules, no build step).

import {
	t,
	initI18n,
	applyI18n,
	localizedName,
	nativeName,
	getUiLanguage,
} from "./i18n.js";

const LS_TOKEN = "mtranserver.token";
const LS_SOURCE = "mtranserver.sourceLang";
const LS_TARGET = "mtranserver.targetLang";
const LS_SOURCE_ORDER = "mtranserver.sourceOrder";
const LS_TARGET_ORDER = "mtranserver.targetOrder";
const AUTO = "auto";

function loadOrder(key) {
	try {
		const raw = localStorage.getItem(key);
		if (!raw) return [];
		const arr = JSON.parse(raw);
		return Array.isArray(arr) ? arr.filter((c) => typeof c === "string") : [];
	} catch {
		return [];
	}
}

const state = {
	sourceLang: AUTO,
	targetLang: "zh-Hans",
	languages: [],
	token: localStorage.getItem(LS_TOKEN) || "",
	uiLanguage: getUiLanguage(),
	requestSeq: 0,
	currentDropdown: null, // "source" | "target"
	dropdownIndex: -1,
	fetched: null, // resolved name for auto detect
	sourceOrder: loadOrder(LS_SOURCE_ORDER),
	targetOrder: loadOrder(LS_TARGET_ORDER),
	hasSavedTarget: false,
	dropdownSnapshot: null, // { side, sourceLang, targetLang }
	clearClicks: [],
};

const el = {
	sourceLangSelect: document.getElementById("sourceLangSelect"),
	targetLangSelect: document.getElementById("targetLangSelect"),
	sourceLangLabel: document.getElementById("sourceLangLabel"),
	targetLangLabel: document.getElementById("targetLangLabel"),
	swapBtn: document.getElementById("swapBtn"),
	sourceText: document.getElementById("sourceText"),
	outputText: document.getElementById("outputText"),
	outputCharCount: document.getElementById("outputCharCount"),
	clearBtn: document.getElementById("clearBtn"),
	srcCopyBtn: document.getElementById("srcCopyBtn"),
	srcSpeakBtn: document.getElementById("srcSpeakBtn"),
	detectedLang: document.getElementById("detectedLang"),
	copyBtn: document.getElementById("copyBtn"),
	speakBtn: document.getElementById("speakBtn"),
	charCount: document.getElementById("charCount"),
	resultStatus: document.getElementById("resultStatus"),
	appVersion: document.getElementById("appVersion"),
	langDropdown: document.getElementById("langDropdown"),
	langSearch: document.getElementById("langSearch"),
	langList: document.getElementById("langList"),
	authModal: document.getElementById("authModal"),
	authTokenInput: document.getElementById("authTokenInput"),
	authOkBtn: document.getElementById("authOkBtn"),
	authCancelBtn: document.getElementById("authCancelBtn"),
	toast: document.getElementById("toast"),
};

/* ------------------------------- API helpers ------------------------------- */

function apiUrl(path) {
	return path;
}

function authHeaders() {
	if (!state.token) return {};
	return { Authorization: `Bearer ${state.token}` };
}

async function apiFetch(path, options = {}) {
	const headers = { ...authHeaders(), ...(options.headers || {}) };
	const res = await fetch(apiUrl(path), { ...options, headers });
	if (res.status === 401) {
		const ok = await promptForToken();
		if (!ok) throw new AuthError();
		return apiFetch(path, options); // retry once with the new token
	}
	return res;
}

class AuthError extends Error {
	constructor() {
		super("auth");
	}
}

async function handleApiError(res, fallbackKey) {
	let msg = t(fallbackKey);
	try {
		const data = await res.json();
		if (data && data.error) msg = String(data.error);
	} catch {
		/* non-JSON error body */
	}
	return msg;
}

/* ------------------------------ Language data ------------------------------ */

async function loadLanguages() {
	const res = await apiFetch("/languages");
	if (!res.ok) {
		throw new Error(await handleApiError(res, "error"));
	}
	const data = await res.json();
	state.languages = data.languages || [];
}

async function loadVersion() {
	try {
		const res = await fetch("/version");
		if (res.ok) {
			const data = await res.json();
			if (data.version) el.appVersion.textContent = `v${data.version}`;
		}
	} catch {
		/* version is cosmetic; ignore */
	}
}

/* --------------------------------- Input ---------------------------------- */

function setResultStatus(text, spinner) {
	if (text == null) {
		el.resultStatus.hidden = true;
		el.resultStatus.textContent = "";
		return;
	}
	el.resultStatus.hidden = false;
	el.resultStatus.textContent = text;
	if (spinner) {
		const spin = document.createElement("span");
		spin.className = "status-spinner";
		el.resultStatus.prepend(spin);
	} else {
		el.resultStatus.querySelector(".status-spinner")?.remove();
	}
}

function updateCharCount() {
	const n = [...el.sourceText.value].length;
	el.charCount.textContent = t("charCount", { n });
	el.clearBtn.hidden = n === 0;
	el.srcCopyBtn.hidden = n === 0;
	el.srcSpeakBtn.hidden = n === 0;
}

function setOutput(text, isPlaceholder) {
	el.outputText.textContent = text;
	el.outputText.classList.toggle("placeholder", !!isPlaceholder);
	const hasOutput = !!text && !isPlaceholder;
	el.copyBtn.hidden = !hasOutput;
	el.speakBtn.hidden = !hasOutput;
	const n = hasOutput ? [...text].length : 0;
	el.outputCharCount.textContent = t("charCount", { n });
	autosize();
}

/* --------------------------------- Translate ------------------------------- */

async function translate(from, to, text) {
	const res = await apiFetch("/translate", {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ from, to, text }),
	});
	if (!res.ok) {
		throw new Error(await handleApiError(res, "error"));
	}
	const data = await res.json();
	return {
		result: data.result || "",
		// Backend reports the detected source language when `from` is "auto";
		// null otherwise (or when detection fails).  Reusing it here avoids a
		// separate /detect round-trip.
		detectedLanguage: data.detected_language || null,
		// Whole-millisecond handling time reported by the backend.
		timeCost: typeof data.time_cost === "number" ? data.time_cost : null,
	};
}

function raceSeq() {
	return ++state.requestSeq;
}

function isStale(seq) {
	return seq !== state.requestSeq;
}

async function runTranslation(seq) {
	const text = el.sourceText.value;
	const from = state.sourceLang;
	const to = state.targetLang;

	if (!text.trim() || text.trim() === "\n") {
		if (!isStale(seq)) {
			state.fetched = null;
			el.detectedLang.textContent = "";
			el.detectedLang.hidden = true;
			setResultStatus(null);
			setOutput(t("noResult"), true);
		}
		return;
	}

	setResultStatus(t("translating"), true);

	// Single /translate request: with `from` set to "auto" the backend both
	// detects the source language and returns it in `detected_language`, so no
	// separate /detect round-trip is needed.
	try {
		const { result, detectedLanguage, timeCost } = await translate(from, to, text);
		if (isStale(seq)) return;

		const detected = from === AUTO ? detectedLanguage : from;
		state.fetched = detected;
		if (from === AUTO && detected) {
			el.detectedLang.textContent = t("detected", {
				lang: localizedName(detected, state.uiLanguage),
			});
			el.detectedLang.hidden = false;
		} else {
			el.detectedLang.textContent = "";
			el.detectedLang.hidden = true;
		}

		setOutput(result);
		setResultStatus(timeCost == null ? t("translationReady") : t("translationReadyTime", { ms: timeCost }));
	} catch (err) {
		if (!isStale(seq)) {
			if (err instanceof AuthError) {
				setOutput("");
				return;
			}
			setOutput("");
			showToast(t("error", { msg: err.message || t("networkError") }), true);
			setResultStatus(null);
		}
	}
}

function scheduleTranslation() {
	const text = el.sourceText.value;
	clearTimeout(state.timer);
	if (!text.trim()) {
		state.requestSeq++; // invalidate any in-flight requests
		state.fetched = null;
		el.detectedLang.textContent = "";
		el.detectedLang.hidden = true;
		if (state.sourceLang === AUTO) {
			el.sourceLangLabel.textContent = t("detect");
		}
		setResultStatus(null);
		setOutput(t("noResult"), true);
		updateCharCount();
		return;
	}
	updateCharCount();
	const seq = raceSeq();
	state.timer = setTimeout(() => runTranslation(seq), 400);
}

/* -------------------------------- Language UI ------------------------------ */

// Default language order by common usage, so the most frequently used
// languages appear at the top of the dropdown.  Chinese (zh-Hans) is first
// per user preference.  Languages not in this list are sorted alphabetically
// after the default entries.
const DEFAULT_LANG_ORDER = [
	"zh-Hans", "zh-Hant",
	"en",
	"es", "fr", "de", "pt", "ru", "ja", "ko", "it", "ar",
	"tr", "vi", "pl", "nl", "el", "th", "hi", "id", "uk",
	"ro", "cs", "sv", "hu", "fi", "da", "bg", "he", "nb",
	"hr", "sr", "sk", "sl", "lt", "lv", "et", "ca", "gl",
	"eu", "fa", "bn", "gu", "kn", "ml", "mr", "pa", "ta",
	"te", "ur", "is", "sq", "ms", "be", "bs", "az", "af",
	"nn", "ne", "si", "sw", "tl",
];

function langEntries(side) {
	const order = side === "source" ? state.sourceOrder : state.targetOrder;
	const defaultRank = new Map(DEFAULT_LANG_ORDER.map((code, i) => [code, i]));
	const recencyRank = new Map(order.map((code, i) => [code, i]));
	return state.languages
		.map((code) => ({ code, local: localizedName(code, state.uiLanguage) }))
		.sort((a, b) => {
			// 1. Recency rank (most recently used first)
			const ra = recencyRank.has(a.code) ? recencyRank.get(a.code) : Infinity;
			const rb = recencyRank.has(b.code) ? recencyRank.get(b.code) : Infinity;
			if (ra !== rb) return ra - rb;
			// 2. Default common-usage rank (Chinese first)
			const da = defaultRank.has(a.code) ? defaultRank.get(a.code) : Infinity;
			const db = defaultRank.has(b.code) ? defaultRank.get(b.code) : Infinity;
			if (da !== db) return da - db;
			// 3. Alphabetical by localized name
			return a.local.localeCompare(b.local, state.uiLanguage);
		});
}

function renderDropDown(side) {
	const query = el.langSearch.value.trim().toLowerCase();
	const active = side === "source" ? state.sourceLang : state.targetLang;
	el.langList.textContent = "";
	state.dropdownIndex = -1;

	// "Auto detect" is always the first item in the source dropdown and never
	// re-ordered by the recency list.
	if (side === "source") {
		const autoItem = document.createElement("div");
		autoItem.className = "dropdown-item" + (active === AUTO ? " active" : "");
		const autoName = document.createElement("span");
		autoName.className = "lang-local-name";
		autoName.textContent = t("detect");
		autoItem.appendChild(autoName);
		autoItem.addEventListener("mousedown", (ev) => {
			ev.preventDefault();
			selectLanguage(side, AUTO);
		});
		el.langList.appendChild(autoItem);
	}

	const entries = langEntries(side).filter(
		(e) =>
			e.local.toLowerCase().includes(query) ||
			nativeName(e.code).toLowerCase().includes(query) ||
			e.code.toLowerCase().includes(query)
	);
	if (!entries.length) {
		const empty = document.createElement("div");
		empty.className = "dropdown-empty";
		empty.textContent = t("searchLang");
		el.langList.appendChild(empty);
		return;
	}
	for (const e of entries) {
		const item = document.createElement("div");
		item.className = "dropdown-item" + (e.code === active ? " active" : "");
		const name = document.createElement("span");
		name.className = "lang-local-name";
		name.textContent = e.local;
		const native = document.createElement("span");
		native.className = "native-name";
		const nativeVal = nativeName(e.code);
		native.textContent = nativeVal || "";
		item.appendChild(name);
		item.appendChild(native);
		item.addEventListener("mousedown", (ev) => {
			ev.preventDefault();
			selectLanguage(side, e.code);
		});
		el.langList.appendChild(item);
	}
}

function openDropDown(side) {
	state.currentDropdown = side;
	state.dropdownSnapshot = {
		side,
		sourceLang: state.sourceLang,
		targetLang: state.targetLang,
	};
	const trigger = side === "source" ? el.sourceLangSelect : el.targetLangSelect;
	const rect = trigger.getBoundingClientRect();
	const dd = el.langDropdown;

	el.langDropdown.hidden = false;
	dd.style.top = `${rect.bottom + 4}px`;
	dd.style.left = `${Math.min(rect.left, window.innerWidth - 340)}px`;
	el.langSearch.value = "";
	trigger.classList.add("active");
	trigger.setAttribute("aria-expanded", "true");
	renderDropDown(side);
	el.langSearch.focus();
}

function closeDropDown() {
	if (!state.currentDropdown) return;
	const trigger = state.currentDropdown === "source" ? el.sourceLangSelect : el.targetLangSelect;
	el.langDropdown.hidden = true;
	trigger.classList.remove("active");
	trigger.setAttribute("aria-expanded", "false");
	state.currentDropdown = null;

	// Commit any confirmed selection to the recency lists only once the
	// dropdown has closed — a picked-but-never-confirmed item must not be
	// recorded. A swap changes both sides, so compare each against the snapshot.
	const snapshot = state.dropdownSnapshot;
	state.dropdownSnapshot = null;
	if (snapshot) {
		if (state.sourceLang !== snapshot.sourceLang && state.sourceLang !== AUTO) {
			touchOrder("source", state.sourceLang);
		}
		if (state.targetLang !== snapshot.targetLang && state.targetLang !== AUTO) {
			touchOrder("target", state.targetLang);
		}
	}
}

function touchOrder(side, code) {
	const key = side === "source" ? LS_SOURCE_ORDER : LS_TARGET_ORDER;
	const list = side === "source" ? state.sourceOrder : state.targetOrder;
	const next = [code, ...list.filter((c) => c !== code)];
	if (side === "source") state.sourceOrder = next;
	else state.targetOrder = next;
	try {
		localStorage.setItem(key, JSON.stringify(next));
	} catch {
		/* storage full/unavailable; recency is cosmetic */
	}
}

function selectLanguage(side, code) {
	const swapSource = side === "source" && state.sourceLang !== AUTO && code === state.targetLang;
	const swapTarget = side === "target" && state.sourceLang !== AUTO && code === state.sourceLang;
	if (swapSource || swapTarget) {
		// Swap the pair instead of forcing duplicates.
		const prevSource = state.sourceLang;
		state.sourceLang = state.targetLang;
		state.targetLang = prevSource;
	} else if (side === "source") {
		state.sourceLang = code;
	} else {
		state.targetLang = code;
	}
	savePrefs();
	closeDropDown();
	refreshLangLabels();
	scheduleTranslation();
}

function refreshLangLabels() {
	if (state.sourceLang === AUTO) {
		el.sourceLangLabel.textContent = t("detect");
	} else {
		el.sourceLangLabel.textContent = localizedName(state.sourceLang, state.uiLanguage);
	}
	el.targetLangLabel.textContent = localizedName(state.targetLang, state.uiLanguage);
}

function savePrefs() {
	if (state.sourceLang !== AUTO) localStorage.setItem(LS_SOURCE, state.sourceLang);
	else localStorage.removeItem(LS_SOURCE);
	localStorage.setItem(LS_TARGET, state.targetLang);
}

/* ---------------------------------- Swap ----------------------------------- */

async function swapLanguages() {
	const oldSource = state.sourceLang;
	const oldTarget = state.targetLang;
	const output = el.outputText.textContent;
	const hasOutput = !!output && !el.outputText.classList.contains("placeholder");
	if (oldSource === AUTO) {
		const detected = state.fetched;
		state.sourceLang = oldTarget;
		state.targetLang = detected && detected !== oldTarget ? detected : "en";
	} else {
		state.sourceLang = oldTarget;
		state.targetLang = oldSource;
	}
	if (hasOutput) {
		el.sourceText.value = output;
		autosize();
	}
	savePrefs();
	if (state.sourceLang !== AUTO) touchOrder("source", state.sourceLang);
	touchOrder("target", state.targetLang);
	refreshLangLabels();
	scheduleTranslation();
}

async function copyToClipboard(text, btn) {
	try {
		await navigator.clipboard.writeText(text);
	} catch {
		// Fallback for insecure contexts.
		const ta = document.createElement("textarea");
		ta.value = text;
		ta.style.position = "fixed";
		ta.style.top = "-9999px";
		document.body.appendChild(ta);
		ta.select();
		document.execCommand("copy");
		ta.remove();
	}
	if (btn) {
		btn.classList.add("flash");
		setTimeout(() => btn.classList.remove("flash"), 1200);
	}
	showToast(t("copied"));
}

function copyOutput() {
	const text = el.outputText.textContent;
	if (!text || el.outputText.classList.contains("placeholder")) return;
	copyToClipboard(text, el.copyBtn);
}

function copySource() {
	const text = el.sourceText.value;
	if (!text) return;
	copyToClipboard(text, el.srcCopyBtn);
}

/* ---------------------------------- Speak ---------------------------------- */

function speakText(text, lang) {
	if (!text) return;
	if (!("speechSynthesis" in window)) return;
	speechSynthesis.cancel();
	const utter = new SpeechSynthesisUtterance(text);
	utter.lang = lang;
	speechSynthesis.speak(utter);
}

function speakOutput() {
	const text = el.outputText.textContent;
	if (!text || el.outputText.classList.contains("placeholder")) return;
	speakText(text, state.targetLang);
}

function speakSource() {
	const text = el.sourceText.value;
	if (!text) return;
	speakText(text, state.sourceLang === AUTO ? (state.fetched || "en") : state.sourceLang);
}

/* ---------------------------------- Toast ---------------------------------- */

let toastTimer = null;

function showToast(msg, isError) {
	el.toast.textContent = msg;
	el.toast.classList.toggle("error", !!isError);
	el.toast.hidden = false;
	clearTimeout(toastTimer);
	toastTimer = setTimeout(() => {
		el.toast.hidden = true;
	}, 2600);
}

/* ---------------------------------- Auth ----------------------------------- */

let _authPrompt = null;

function promptForToken() {
	if (_authPrompt) return _authPrompt;
	_authPrompt = new Promise((resolve) => {
		el.authTokenInput.value = state.token;
		el.authModal.hidden = false;
		el.authTokenInput.focus();

		const done = (ok) => {
			el.authModal.hidden = true;
			_authPrompt = null;
			resolve(ok);
		};

		el.authOkBtn.onclick = () => {
			const val = el.authTokenInput.value.trim();
			state.token = val;
			if (val) localStorage.setItem(LS_TOKEN, val);
			else localStorage.removeItem(LS_TOKEN);
			showToast(t("saveToken"));
			done(!!val);
		};
		el.authCancelBtn.onclick = () => done(false);
		el.authTokenInput.onkeydown = (ev) => {
			if (ev.key === "Enter") el.authOkBtn.click();
			if (ev.key === "Escape") el.authCancelBtn.click();
		};
	});
	return _authPrompt;
}

/* ------------------------------- Autosize textarea -------------------------- */

function autosize() {
	const prevScrollY = window.scrollY;
	el.sourceText.style.height = "auto";
	el.sourceText.style.minHeight = "";
	const h = el.sourceText.scrollHeight;
	el.sourceText.style.height = `${h}px`;
	el.sourceText.style.minHeight = `${h}px`;
	if (window.scrollY !== prevScrollY) {
		window.scrollTo(window.scrollX, prevScrollY);
	}
}

function initialAutoSelectTarget() {
	// No saved target preference: default to the browser language when the
	// model supports it, otherwise fall back to English.
	if (state.hasSavedTarget) return;
	const ui = state.uiLanguage;
	let candidate;
	if (state.languages.includes(ui)) {
		candidate = ui;
	} else {
		const base = ui.split("-")[0];
		candidate = state.languages.includes(base) ? base : "en";
	}
	// Never default to the same language as a fixed source.
	if (state.sourceLang !== AUTO && candidate === state.sourceLang) {
		candidate = "en";
	}
	state.targetLang = candidate;
}

/* ---------------------------------- Events --------------------------------- */

function moveDropdownIndex(next) {
	const items = el.langList.querySelectorAll(".dropdown-item");
	if (!items.length) {
		state.dropdownIndex = -1;
		return;
	}
	state.dropdownIndex = (next + items.length) % items.length;
	items.forEach((item, i) => {
		item.classList.toggle("focused", i === state.dropdownIndex);
	});
	items[state.dropdownIndex].scrollIntoView({ block: "nearest" });
}

function bindEvents() {
	el.sourceText.addEventListener("input", () => {
		scheduleTranslation();
		autosize();
	});
	el.sourceText.addEventListener("paste", () => {
		requestAnimationFrame(autosize);
	});
	el.sourceText.addEventListener("cut", () => {
		requestAnimationFrame(autosize);
	});
	el.sourceText.addEventListener("keydown", (ev) => {
		if ((ev.ctrlKey || ev.metaKey) && ev.key === "Enter") {
			ev.preventDefault();
			clearTimeout(state.timer);
			runTranslation(raceSeq());
		}
	});

	el.clearBtn.addEventListener("click", () => {
		const now = Date.now();
		state.clearClicks = (el.sourceText.value === "" ? state.clearClicks : []).filter((ts) => now - ts < 1000);
		if (state.clearClicks.push(now) >= 3) {
			state.clearClicks = [];
			const text = "I Love the World and Everything In It";
			let i = 0;
			const step = () => {
				el.sourceText.value = text.slice(0, ++i);
				autosize();
				updateCharCount();
				i < text.length ? setTimeout(step, 35) : scheduleTranslation();
			};
			setTimeout(step, 100);
		} else {
			el.sourceText.value = "";
			autosize();
			scheduleTranslation();
			el.sourceText.focus();
		}
	});
	el.copyBtn.addEventListener("click", copyOutput);
	el.speakBtn.addEventListener("click", speakOutput);
	el.srcCopyBtn.addEventListener("click", copySource);
	el.srcSpeakBtn.addEventListener("click", speakSource);

	el.swapBtn.addEventListener("click", swapLanguages);

	el.sourceLangSelect.addEventListener("click", () => {
		if (state.currentDropdown === "source") closeDropDown();
		else {
			closeDropDown();
			openDropDown("source");
		}
	});
	el.sourceLangSelect.addEventListener("keydown", (ev) => {
		if (ev.key === "Enter" || ev.key === "ArrowDown") {
			ev.preventDefault();
			closeDropDown();
			openDropDown("source");
		}
		if (ev.key === "Escape") closeDropDown();
	});

	el.targetLangSelect.addEventListener("click", () => {
		if (state.currentDropdown === "target") closeDropDown();
		else {
			closeDropDown();
			openDropDown("target");
		}
	});
	el.targetLangSelect.addEventListener("keydown", (ev) => {
		if (ev.key === "Enter" || ev.key === "ArrowDown") {
			ev.preventDefault();
			closeDropDown();
			openDropDown("target");
		}
		if (ev.key === "Escape") closeDropDown();
	});

	el.langSearch.addEventListener("input", () => {
		renderDropDown(state.currentDropdown);
		moveDropdownIndex(0);
	});
	el.langSearch.addEventListener("keydown", (ev) => {
		const items = el.langList.querySelectorAll(".dropdown-item");
		if (ev.key === "ArrowDown") {
			ev.preventDefault();
			moveDropdownIndex(state.dropdownIndex + 1);
		} else if (ev.key === "ArrowUp") {
			ev.preventDefault();
			moveDropdownIndex(state.dropdownIndex - 1);
		} else if (ev.key === "Enter") {
			ev.preventDefault();
			const target = items[state.dropdownIndex];
			if (target) target.dispatchEvent(new MouseEvent("mousedown", { bubbles: true }));
		} else if (ev.key === "Escape") {
			closeDropDown();
		}
	});

	document.addEventListener("mousedown", (ev) => {
		if (
			state.currentDropdown &&
			!el.langDropdown.contains(ev.target) &&
			!el.sourceLangSelect.contains(ev.target) &&
			!el.targetLangSelect.contains(ev.target)
		) {
			closeDropDown();
		}
	});

	window.addEventListener("resize", () => {
		autosize();
		if (state.currentDropdown) {
			const trigger = state.currentDropdown === "source" ? el.sourceLangSelect : el.targetLangSelect;
			const rect = trigger.getBoundingClientRect();
			el.langDropdown.style.top = `${rect.bottom + 4}px`;
			el.langDropdown.style.left = `${Math.min(rect.left, window.innerWidth - 340)}px`;
		}
	});

	// Auth modal: keep focus inside and allow Escape from anywhere in the modal.
	el.authModal.addEventListener("keydown", (ev) => {
		if (ev.key === "Escape") {
			ev.preventDefault();
			el.authCancelBtn.click();
			return;
		}
		if (ev.key !== "Tab") return;
		const focusables = [el.authTokenInput, el.authCancelBtn, el.authOkBtn];
		const idx = focusables.indexOf(document.activeElement);
		if (idx === -1) {
			ev.preventDefault();
			el.authTokenInput.focus();
			return;
		}
		ev.preventDefault();
		const next = ev.shiftKey
			? (idx - 1 + focusables.length) % focusables.length
			: (idx + 1) % focusables.length;
		focusables[next].focus();
	});
}

/* ---------------------------------- Init ----------------------------------- */

async function init() {
	initI18n();
	state.uiLanguage = getUiLanguage();
	bindEvents();
	applyI18n();

	// Pick up token from URL query parameter.
	const urlParams = new URLSearchParams(location.search);
	const urlToken = urlParams.get("token");
	if (urlToken) {
		state.token = urlToken;
		localStorage.setItem(LS_TOKEN, urlToken);
	}

	// Restore preferences.
	const savedSource = localStorage.getItem(LS_SOURCE);
	const savedTarget = localStorage.getItem(LS_TARGET);
	state.hasSavedTarget = !!savedTarget;
	if (savedSource && savedSource !== AUTO) state.sourceLang = savedSource;
	if (savedTarget) state.targetLang = savedTarget;
	// Ensure source != target on restore.
	if (state.sourceLang !== AUTO && state.sourceLang === state.targetLang) {
		state.targetLang = state.sourceLang === "zh-Hans" ? "en" : "zh-Hans";
	}

	refreshLangLabels();
	updateCharCount();
	setOutput(t("noResult"), true);
	autosize();

	loadVersion();

	try {
		await loadLanguages();
		initialAutoSelectTarget();
		refreshLangLabels();
	} catch (err) {
		if (err instanceof AuthError) return; // user interaction handled in modal
		showToast(t("error", { msg: err.message }), true);
	}
}

init();
