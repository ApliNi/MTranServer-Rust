// MTranServer-Rust UI — automatic i18n. Picks the first supported browser language,
// no manual selection. Unknown languages fall back to English.

const SUPPORTED = [
	"zh-Hans", "zh-Hant", "en", "ja", "ko", "fr", "de", "es", "pt", "ru",
	"it", "tr", "uk", "vi", "th", "id", "ar", "hi",
];

const MESSAGES = {
	"zh-Hans": {
		"placeholder": "输入文本",
		"swap": "交换语言",
		"clear": "清空",
		"copy": "复制译文",
		"copied": "已复制",
		"listen": "朗读",
		"detect": "自动识别",
		"detected": "已检测为 {lang}",
		"translating": "正在翻译…",
		"translationReady": "译文就绪",
		"translationReadyTime": "译文就绪, 耗时 {ms}ms",
		"searchLang": "搜索语言",
		"searchPlaceholder": "搜索语言",
		"authTitle": "需要身份验证",
		"authDesc": "此服务器需要 API 令牌，请输入后继续。",
		"authCancel": "取消",
		"authSave": "保存",
		"saveToken": "令牌已保存",
		"error": "翻译失败：{msg}",
		"networkError": "无法连接到服务器",
		"noResult": "译文将显示在此处",
		"charCount": "{n} 字符",
	},
	"zh-Hant": {
		"placeholder": "輸入文字",
		"swap": "交換語言",
		"clear": "清除",
		"copy": "複製譯文",
		"copied": "已複製",
		"listen": "朗讀",
		"detect": "自動識別",
		"detected": "已偵測為 {lang}",
		"translating": "正在翻譯…",
		"translationReady": "譯文就緒",
		"translationReadyTime": "譯文就緒, 耗時 {ms}ms",
		"searchLang": "搜尋語言",
		"searchPlaceholder": "搜尋語言",
		"authTitle": "需要身分驗證",
		"authDesc": "此伺服器需要 API 令牌，請輸入後繼續。",
		"authCancel": "取消",
		"authSave": "儲存",
		"saveToken": "令牌已儲存",
		"error": "翻譯失敗：{msg}",
		"networkError": "無法連線到伺服器",
		"noResult": "譯文將顯示在此處",
		"charCount": "{n} 字元",
	},
	en: {
		"placeholder": "Enter text",
		"swap": "Swap languages",
		"clear": "Clear",
		"copy": "Copy translation",
		"copied": "Copied",
		"listen": "Listen",
		"detect": "Auto detect",
		"detected": "Detected: {lang}",
		"translating": "Translating…",
		"translationReady": "Translation ready",
		"translationReadyTime": "Translation ready, {ms}ms",
		"searchLang": "Search languages",
		"searchPlaceholder": "Search languages",
		"authTitle": "Authentication required",
		"authDesc": "This server requires an API token. Enter it to continue.",
		"authCancel": "Cancel",
		"authSave": "Save",
		"saveToken": "Token saved",
		"error": "Translation failed: {msg}",
		"networkError": "Cannot connect to the server",
		"noResult": "Translation will appear here",
		"charCount": "{n} characters",
	},
	ja: {
		"placeholder": "テキストを入力",
		"swap": "言語を入れ替え",
		"clear": "クリア",
		"copy": "翻訳をコピー",
		"copied": "コピーしました",
		"listen": "読み上げ",
		"detect": "自動検出",
		"detected": "検出: {lang}",
		"translating": "翻訳中…",
		"translationReady": "翻訳完了",
		"translationReadyTime": "翻訳完了, 所要時間 {ms}ms",
		"searchLang": "言語を検索",
		"searchPlaceholder": "言語を検索",
		"authTitle": "認証が必要です",
		"authDesc": "このサーバーには API トークンが必要です。入力して続行してください。",
		"authCancel": "キャンセル",
		"authSave": "保存",
		"saveToken": "トークンを保存しました",
		"error": "翻訳に失敗: {msg}",
		"networkError": "サーバーに接続できません",
		"noResult": "翻訳はここに表示されます",
		"charCount": "{n} 文字",
	},
	ko: {
		"placeholder": "텍스트 입력",
		"swap": "언어 바꾸기",
		"clear": "지우기",
		"copy": "번역 복사",
		"copied": "복사됨",
		"listen": "듣기",
		"detect": "자동 감지",
		"detected": "감지됨: {lang}",
		"translating": "번역 중…",
		"translationReady": "번역 완료",
		"translationReadyTime": "번역 완료, 소요 {ms}ms",
		"searchLang": "언어 검색",
		"searchPlaceholder": "언어 검색",
		"authTitle": "인증 필요",
		"authDesc": "이 서버에는 API 토큰이 필요합니다. 입력하여 계속하세요.",
		"authCancel": "취소",
		"authSave": "저장",
		"saveToken": "토큰이 저장되었습니다",
		"error": "번역 실패: {msg}",
		"networkError": "서버에 연결할 수 없습니다",
		"noResult": "번역이 여기에 표시됩니다",
		"charCount": "{n}자",
	},
	fr: {
		"placeholder": "Saisissez du texte",
		"swap": "Inverser les langues",
		"clear": "Effacer",
		"copy": "Copier la traduction",
		"copied": "Copié",
		"listen": "Écouter",
		"detect": "Détection automatique",
		"detected": "Détecté : {lang}",
		"translating": "Traduction…",
		"translationReady": "Traduction prête",
		"translationReadyTime": "Traduction prête, {ms} ms",
		"searchLang": "Rechercher des langues",
		"searchPlaceholder": "Rechercher des langues",
		"authTitle": "Authentification requise",
		"authDesc": "Ce serveur nécessite un jeton API. Saisissez-le pour continuer.",
		"authCancel": "Annuler",
		"authSave": "Enregistrer",
		"saveToken": "Jeton enregistré",
		"error": "Échec de la traduction : {msg}",
		"networkError": "Impossible de se connecter au serveur",
		"noResult": "La traduction apparaîtra ici",
		"charCount": "{n} caractères",
	},
	de: {
		"placeholder": "Text eingeben",
		"swap": "Sprachen tauschen",
		"clear": "Löschen",
		"copy": "Übersetzung kopieren",
		"copied": "Kopiert",
		"listen": "Anhören",
		"detect": "Automatisch erkennen",
		"detected": "Erkannt: {lang}",
		"translating": "Übersetzung…",
		"translationReady": "Übersetzung fertig",
		"translationReadyTime": "Übersetzung fertig, {ms} ms",
		"searchLang": "Sprachen suchen",
		"searchPlaceholder": "Sprachen suchen",
		"authTitle": "Authentifizierung erforderlich",
		"authDesc": "Dieser Server benötigt ein API-Token. Geben Sie es ein, um fortzufahren.",
		"authCancel": "Abbrechen",
		"authSave": "Speichern",
		"saveToken": "Token gespeichert",
		"error": "Übersetzung fehlgeschlagen: {msg}",
		"networkError": "Keine Verbindung zum Server möglich",
		"noResult": "Die Übersetzung erscheint hier",
		"charCount": "{n} Zeichen",
	},
	es: {
		"placeholder": "Introduce texto",
		"swap": "Intercambiar idiomas",
		"clear": "Borrar",
		"copy": "Copiar traducción",
		"copied": "Copiado",
		"listen": "Escuchar",
		"detect": "Detección automática",
		"detected": "Detectado: {lang}",
		"translating": "Traduciendo…",
		"translationReady": "Traducción lista",
		"translationReadyTime": "Traducción lista, {ms} ms",
		"searchLang": "Buscar idiomas",
		"searchPlaceholder": "Buscar idiomas",
		"authTitle": "Autenticación requerida",
		"authDesc": "Este servidor requiere un token de API. Introdúcelo para continuar.",
		"authCancel": "Cancelar",
		"authSave": "Guardar",
		"saveToken": "Token guardado",
		"error": "Error de traducción: {msg}",
		"networkError": "No se puede conectar con el servidor",
		"noResult": "La traducción aparecerá aquí",
		"charCount": "{n} caracteres",
	},
	pt: {
		"placeholder": "Digite o texto",
		"swap": "Inverter idiomas",
		"clear": "Limpar",
		"copy": "Copiar tradução",
		"copied": "Copiado",
		"listen": "Ouvir",
		"detect": "Detección automática",
		"detected": "Detectado: {lang}",
		"translating": "Traduzindo…",
		"translationReady": "Tradução pronta",
		"translationReadyTime": "Tradução pronta, {ms} ms",
		"searchLang": "Pesquisar idiomas",
		"searchPlaceholder": "Pesquisar idiomas",
		"authTitle": "Autenticação necessária",
		"authDesc": "Este servidor requer um token de API. Digite-o para continuar.",
		"authCancel": "Cancelar",
		"authSave": "Salvar",
		"saveToken": "Token salvo",
		"error": "Falha na tradução: {msg}",
		"networkError": "Não foi possível conectar ao servidor",
		"noResult": "A tradução aparecerá aqui",
		"charCount": "{n} caracteres",
	},
	ru: {
		"placeholder": "Введите текст",
		"swap": "Поменять языки",
		"clear": "Очистить",
		"copy": "Копировать перевод",
		"copied": "Скопировано",
		"listen": "Слушать",
		"detect": "Автоопределение",
		"detected": "Определено: {lang}",
		"translating": "Перевод…",
		"translationReady": "Перевод готов",
		"translationReadyTime": "Перевод готов, {ms} мс",
		"searchLang": "Поиск языков",
		"searchPlaceholder": "Поиск языков",
		"authTitle": "Требуется аутентификация",
		"authDesc": "Для этого сервера требуется API-токен. Введите его, чтобы продолжить.",
		"authCancel": "Отмена",
		"authSave": "Сохранить",
		"saveToken": "Токен сохранён",
		"error": "Ошибка перевода: {msg}",
		"networkError": "Не удалось подключиться к серверу",
		"noResult": "Перевод появится здесь",
		"charCount": "{n} символов",
	},
	it: {
		"placeholder": "Inserisci il testo",
		"swap": "Scambia lingue",
		"clear": "Cancella",
		"copy": "Copia traduzione",
		"copied": "Copiato",
		"listen": "Ascolta",
		"detect": "Rilevamento automatico",
		"detected": "Rilevato: {lang}",
		"translating": "Traduzione…",
		"translationReady": "Traduzione pronta",
		"translationReadyTime": "Traduzione pronta, {ms} ms",
		"searchLang": "Cerca lingue",
		"searchPlaceholder": "Cerca lingue",
		"authTitle": "Autenticazione richiesta",
		"authDesc": "Questo server richiede un token API. Inseriscilo per continuare.",
		"authCancel": "Annulla",
		"authSave": "Salva",
		"saveToken": "Token salvato",
		"error": "Errore di traduzione: {msg}",
		"networkError": "Impossibile connettersi al server",
		"noResult": "La traduzione apparirà qui",
		"charCount": "{n} caratteri",
	},
	tr: {
		"placeholder": "Metin girin",
		"swap": "Dilleri değiştir",
		"clear": "Temizle",
		"copy": "Çeviriyi kopyala",
		"copied": "Kopyalandı",
		"listen": "Dinle",
		"detect": "Otomatik algıla",
		"detected": "Algılandı: {lang}",
		"translating": "Çevriliyor…",
		"translationReady": "Çeviri hazır",
		"translationReadyTime": "Çeviri hazır, {ms} ms",
		"searchLang": "Dil ara",
		"searchPlaceholder": "Dil ara",
		"authTitle": "Kimlik doğrulama gerekli",
		"authDesc": "Bu sunucu bir API belirteci gerektirir. Devam etmek için girin.",
		"authCancel": "İptal",
		"authSave": "Kaydet",
		"saveToken": "Belirteç kaydedildi",
		"error": "Çeviri başarısız: {msg}",
		"networkError": "Sunucuya bağlanılamıyor",
		"noResult": "Çeviri burada görünecek",
		"charCount": "{n} karakter",
	},
	uk: {
		"placeholder": "Введіть текст",
		"swap": "Поміняти мови",
		"clear": "Очистити",
		"copy": "Скопіювати переклад",
		"copied": "Скопійовано",
		"listen": "Слухати",
		"detect": "Автовизначення",
		"detected": "Визначено: {lang}",
		"translating": "Переклад…",
		"translationReady": "Переклад готовий",
		"translationReadyTime": "Переклад готовий, {ms} мс",
		"searchLang": "Пошук мов",
		"searchPlaceholder": "Пошук мов",
		"authTitle": "Потрібна автентифікація",
		"authDesc": "Цей сервер потребує токен API. Введіть його, щоб продовжити.",
		"authCancel": "Скасувати",
		"authSave": "Зберегти",
		"saveToken": "Токен збережено",
		"error": "Помилка перекладу: {msg}",
		"networkError": "Не вдалося підключитися до сервера",
		"noResult": "Переклад з’явиться тут",
		"charCount": "{n} символів",
	},
	vi: {
		"placeholder": "Nhập văn bản",
		"swap": "Đổi ngôn ngữ",
		"clear": "Xóa",
		"copy": "Sao chép bản dịch",
		"copied": "Đã sao chép",
		"listen": "Nghe",
		"detect": "Tự động phát hiện",
		"detected": "Đã phát hiện: {lang}",
		"translating": "Đang dịch…",
		"translationReady": "Bản dịch sẵn sàng",
		"translationReadyTime": "Bản dịch sẵn sàng, {ms}ms",
		"searchLang": "Tìm ngôn ngữ",
		"searchPlaceholder": "Tìm ngôn ngữ",
		"authTitle": "Cần xác thực",
		"authDesc": "Máy chủ này yêu cầu mã thông báo API. Nhập mã để tiếp tục.",
		"authCancel": "Hủy",
		"authSave": "Lưu",
		"saveToken": "Đã lưu mã thông báo",
		"error": "Dịch thất bại: {msg}",
		"networkError": "Không thể kết nối đến máy chủ",
		"noResult": "Bản dịch sẽ hiển thị tại đây",
		"charCount": "{n} ký tự",
	},
	id: {
		"placeholder": "Masukkan teks",
		"swap": "Tukar bahasa",
		"clear": "Hapus",
		"copy": "Salin terjemahan",
		"copied": "Disalin",
		"listen": "Dengarkan",
		"detect": "Deteksi otomatis",
		"detected": "Terdeteksi: {lang}",
		"translating": "Menerjemahkan…",
		"translationReady": "Terjemahan siap",
		"translationReadyTime": "Terjemahan siap, {ms}ms",
		"searchLang": "Cari bahasa",
		"searchPlaceholder": "Cari bahasa",
		"authTitle": "Diperlukan autentikasi",
		"authDesc": "Server ini memerlukan token API. Masukkan untuk melanjutkan.",
		"authCancel": "Batal",
		"authSave": "Simpan",
		"saveToken": "Token disimpan",
		"error": "Terjemahan gagal: {msg}",
		"networkError": "Tidak dapat terhubung ke server",
		"noResult": "Terjemahan akan muncul di sini",
		"charCount": "{n} karakter",
	},
	th: {
		"placeholder": "ป้อนข้อความ",
		"swap": "สลับภาษา",
		"clear": "ล้าง",
		"copy": "คัดลอกคำแปล",
		"copied": "คัดลอกแล้ว",
		"listen": "ฟัง",
		"detect": "ตรวจจับอัตโนมัติ",
		"detected": "ตรวจพบ: {lang}",
		"translating": "กำลังแปล…",
		"translationReady": "คำแปลพร้อมแล้ว",
		"translationReadyTime": "คำแปลพร้อมแล้ว, {ms}ms",
		"searchLang": "ค้นหาภาษา",
		"searchPlaceholder": "ค้นหาภาษา",
		"authTitle": "ต้องมีการยืนยันตัวตน",
		"authDesc": "เซิร์ฟเวอร์นี้ต้องใช้โทเค็น API ป้อนเพื่อดำเนินการต่อ",
		"authCancel": "ยกเลิก",
		"authSave": "บันทึก",
		"saveToken": "บันทึกโทเค็นแล้ว",
		"error": "การแปลล้มเหลว: {msg}",
		"networkError": "ไม่สามารถเชื่อมต่อกับเซิร์ฟเวอร์ได้",
		"noResult": "คำแปลจะแสดงที่นี่",
		"charCount": "{n} ตัวอักษร",
	},
	ar: {
		"placeholder": "أدخل النص",
		"swap": "تبديل اللغات",
		"clear": "مسح",
		"copy": "نسخ الترجمة",
		"copied": "تم النسخ",
		"listen": "استماع",
		"detect": "الكشف التلقائي",
		"detected": "تم الكشف: {lang}",
		"translating": "جارٍ الترجمة…",
		"translationReady": "الترجمة جاهزة",
		"translationReadyTime": "الترجمة جاهزة, {ms}ms",
		"searchLang": "البحث عن اللغات",
		"searchPlaceholder": "البحث عن اللغات",
		"authTitle": "مصادقة مطلوبة",
		"authDesc": "يتطلب هذا الخادم رمز API. أدخله للمتابعة.",
		"authCancel": "إلغاء",
		"authSave": "حفظ",
		"saveToken": "تم حفظ الرمز",
		"error": "فشلت الترجمة: {msg}",
		"networkError": "تعذر الاتصال بالخادم",
		"noResult": "ستظهر الترجمة هنا",
		"charCount": "{n} حرف",
	},
	hi: {
		"placeholder": "टेक्स्ट दर्ज करें",
		"swap": "भाषाएँ बदलें",
		"clear": "साफ़ करें",
		"copy": "अनुवाद कॉपी करें",
		"copied": "कॉपी हो गया",
		"listen": "सुनें",
		"detect": "स्वतः पहचान",
		"detected": "पहचाना गया: {lang}",
		"translating": "अनुवाद हो रहा है…",
		"translationReady": "अनुवाद तैयार",
		"translationReadyTime": "अनुवाद तैयार, {ms}ms",
		"searchLang": "भाषाएँ खोजें",
		"searchPlaceholder": "भाषाएँ खोजें",
		"authTitle": "प्रमाणीकरण आवश्यक",
		"authDesc": "इस सर्वर के लिए API टोकन आवश्यक है। जारी रखने के लिए दर्ज करें।",
		"authCancel": "रद्द करें",
		"authSave": "सहेजें",
		"saveToken": "टोकन सहेजा गया",
		"error": "अनुवाद विफल: {msg}",
		"networkError": "सर्वर से कनेक्ट नहीं हो सका",
		"noResult": "अनुवाद यहाँ दिखाई देगा",
		"charCount": "{n} वर्ण",
	},
};

// Cache for language-name display (in UI language and native).
const _displayNames = new Map();

function displayNamesFor(locale) {
	if (!_displayNames.has(locale)) {
		try {
			_displayNames.set(locale, new Intl.DisplayNames([locale], { type: "language" }));
		} catch {
			_displayNames.set(locale, new Intl.DisplayNames(["en"], { type: "language" }));
		}
	}
	return _displayNames.get(locale);
}

export function detectUiLanguage() {
	const preferred = navigator.languages && navigator.languages.length
		? navigator.languages
		: [navigator.language || "en"];

	for (const raw of preferred) {
		const normalized = normalizeLocale(raw);
		// Exact match first, then base-language match.
		if (SUPPORTED.includes(normalized)) return normalized;
		const base = normalized.split("-")[0];
		const match = SUPPORTED.find((l) => l === base || l.startsWith(base + "-"));
		if (match) return match;
	}
	return "en";
}

export function normalizeLocale(locale) {
	return locale
		.replace(/_/g, "-")
		.replace(/^zh-CN$/i, "zh-Hans")
		.replace(/^zh-TW$/i, "zh-Hant")
		.replace(/^zh-HK$/i, "zh-Hant")
		.replace(/^zh-MO$/i, "zh-Hant")
		.replace(/^zh-SG$/i, "zh-Hans");
}

export function t(key, vars) {
	const table = MESSAGES[currentLang] || MESSAGES.en;
	let msg = table[key] ?? MESSAGES.en[key] ?? key;
	if (vars) {
		for (const [k, v] of Object.entries(vars)) {
			msg = msg.replaceAll(`{${k}}`, String(v));
		}
	}
	return msg;
}

// Language code -> display name in the UI language.
export function localizedName(code, locale) {
	try {
		return displayNamesFor(locale).of(code);
	} catch {
		return code;
	}
}

// Language code -> name written in its own language.
export function nativeName(code) {
	try {
		return displayNamesFor(code).of(code) || "";
	} catch {
		return "";
	}
}

let currentLang = "en";

export function getUiLanguage() {
	return currentLang;
}

export function applyI18n() {
	document.querySelectorAll("[data-i18n]").forEach((el) => {
		el.textContent = t(el.dataset.i18n);
	});
	document.querySelectorAll("[data-i18n-title]").forEach((el) => {
		el.title = t(el.dataset.i18nTitle);
	});
	document.querySelectorAll("[data-i18n-aria]").forEach((el) => {
		el.setAttribute("aria-label", t(el.dataset.i18nAria));
	});
	document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
		el.placeholder = t(el.dataset.i18nPlaceholder);
	});
}

export function initI18n() {
	currentLang = detectUiLanguage();
	document.documentElement.lang = currentLang;
	applyI18n();
	return currentLang;
}
