use parking_lot::Mutex;
use std::fmt::Write as _;
use std::fs;
use std::io::{BufWriter, Write as _};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use std::sync::mpsc;
use std::sync::OnceLock;

use crate::config;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Debug = 0,
    Info = 1,
    Warn = 2,
    Error = 3,
}

impl LogLevel {
    fn from_str(s: &str) -> LogLevel {
        match s.to_lowercase().as_str() {
            "debug" => LogLevel::Debug,
            "info" => LogLevel::Info,
            "warn" => LogLevel::Warn,
            "error" => LogLevel::Error,
            _ => {
                eprintln!(
                    "[WARN] config: invalid MT_LOG_LEVEL `{}`, using \"warn\"",
                    s
                );
                LogLevel::Warn
            }
        }
    }

    fn label(&self) -> &'static str {
        match self {
            LogLevel::Debug => "DBUG",
            LogLevel::Info => "INFO",
            LogLevel::Warn => "WARN",
            LogLevel::Error => "EROR",
        }
    }

    fn color(&self) -> &'static str {
        match self {
            LogLevel::Debug => "\x1b[36m",
            LogLevel::Info => "\x1b[32m",
            LogLevel::Warn => "\x1b[33m",
            LogLevel::Error => "\x1b[31m",
        }
    }
}

static LOG_STATE: Mutex<Option<LogState>> = Mutex::new(None);

struct LogState {
    level: LogLevel,
}

/// A fully formatted log line plus its pre-rendered colored variant.
///
/// `flush` is a sentinel: the writer thread flushes its buffered file output
/// when it receives one, without stopping. `quit` is a sentinel that makes
/// the writer thread drain and return; it is only sent from `shutdown()`.
struct LogLine {
    plain: String,
    colored: Option<String>,
    flush: bool,
    quit: bool,
}

static LOG_SENDER: OnceLock<mpsc::SyncSender<LogLine>> = OnceLock::new();
static WRITER_HANDLE: parking_lot::Mutex<Option<std::thread::JoinHandle<()>>> =
    parking_lot::Mutex::new(None);

/// Initialize the logger from the global config.
pub fn init() {
    let cfg = config::get();
    let level = LogLevel::from_str(&cfg.log_level);

    let file = if cfg.log_to_file {
        fs::create_dir_all(&cfg.log_dir).ok();
        clean_expired_logs(&cfg.log_dir, cfg.log_retention_days);
        open_log_file(&cfg.log_dir)
    } else {
        None
    };

    let (tx, rx) = mpsc::sync_channel::<LogLine>(10_000);
    if let Ok(handle) = std::thread::Builder::new()
        .name("logger".to_string())
        .spawn(move || writer_loop(rx, file))
    {
        LOG_SENDER.set(tx).ok();
        *WRITER_HANDLE.lock() = Some(handle);
    }

    *LOG_STATE.lock() = Some(LogState { level });

    install_tracing_bridge(level);
}

/// Open today's log file, named by UTC date (`mtran-YYYY-MM-DD.log`).
fn log_filename_for_date(year: i64, month: u32, day: u32) -> String {
    format!("mtran-{:04}-{:02}-{:02}.log", year, month, day)
}

fn open_log_file(log_dir: &std::path::Path) -> Option<fs::File> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let (y, m, d) = civil_from_days(now.as_secs());
    let path = log_dir.join(log_filename_for_date(y, m, d));
    fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .ok()
}

/// Drain the log channel forever: write console lines, append to the daily
/// file, rotate when the UTC day changes. Exits on the `quit` sentinel.
fn writer_loop(rx: mpsc::Receiver<LogLine>, file: Option<fs::File>) {
    let mut writer = file.map(|f| std::io::BufWriter::with_capacity(8 * 1024, f));
    // UTC day of the file currently held in `writer`; reopening happens when
    // the day changes so logs rotate at midnight without dropping lines.
    let mut file_date = current_date();
    while let Ok(line) = rx.recv() {
        if line.quit {
            break;
        }
        let day_changed = writer.is_some() && file_date != current_date();
        if let Some(colored) = line.colored {
            println!("{}", colored);
        }
        // Flush before writing whenever a flush was requested or the file is
        // about to be swapped for the new day.
        if line.flush || day_changed {
            if let Some(w) = writer.as_mut() {
                let _ = w.flush();
            }
        }
        if day_changed {
            file_date = current_date();
            clean_expired_logs(&config::get().log_dir, config::get().log_retention_days);
            writer = open_log_file(&config::get().log_dir).map(BufWriter::new);
        }
        if let Some(w) = writer.as_mut() {
            let _ = writeln!(w, "{}", line.plain);
            if line.flush {
                let _ = w.flush();
            }
        }
    }
    if let Some(mut w) = writer {
        let _ = w.flush();
    }
}

/// Current UTC day as `(year, month, day)` for log-file naming/rotation.
fn current_date() -> (i64, u32, u32) {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    civil_from_days(now.as_secs())
}

/// Flush buffered file output and stop the writer thread. Called on the
/// graceful-shutdown path before the process exits: statics are not dropped
/// at process exit, so the BufWriter tail would be lost otherwise.
///
/// The exit sentinel is enqueued through the same channel, so every line sent
/// before it is written first; a short bounded wait covers the writer's final
/// flush before the process moves on.
pub fn shutdown() {
    if let Some(tx) = LOG_SENDER.get() {
        let _ = tx.send(LogLine {
            plain: String::new(),
            colored: None,
            flush: true,
            quit: true,
        });
    }
    if let Some(handle) = WRITER_HANDLE.lock().take() {
        let _ = handle.join();
    }
}

/// Bridge `tracing` events (used by axum/tower middleware) into this custom
/// logger so request logs and other tracing output appear in the same stream
/// with the same timestamp/level formatting as the TS version.
fn install_tracing_bridge(min_level: LogLevel) {
    let filter = tracing_subscriber::EnvFilter::new(match min_level {
        LogLevel::Debug => "debug",
        LogLevel::Info => "info",
        LogLevel::Warn => "warn",
        LogLevel::Error => "error",
    });

    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(BridgeLayer)
        .try_init();
}

/// A `tracing` layer that forwards events to the custom logger, matching the
/// TS `logger.*` output format (`[LEVEL] YYYY-MM-DD HH:MM:SS.mmm message`).
struct BridgeLayer;

struct MessageVisitor<'a> {
    message: &'a mut String,
}

impl<'a> tracing::field::Visit for MessageVisitor<'a> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            let _ = write!(self.message, "{:?}", value);
        }
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "message" {
            self.message.push_str(value);
        }
    }
}

impl<S> tracing_subscriber::Layer<S> for BridgeLayer
where
    S: tracing::Subscriber,
{
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let mut message = String::new();
        let mut visitor = MessageVisitor {
            message: &mut message,
        };
        event.record(&mut visitor);

        if message.is_empty() {
            return;
        }

        let level = match *event.metadata().level() {
            tracing::Level::ERROR => LogLevel::Error,
            tracing::Level::WARN => LogLevel::Warn,
            tracing::Level::INFO => LogLevel::Info,
            tracing::Level::DEBUG | tracing::Level::TRACE => LogLevel::Debug,
        };

        log(level, &message);
    }
}

/// Convert days since the Unix epoch to `(year, month, day)` in UTC.
///
/// Howard Hinnant's `civil_from_days` algorithm — proleptic Gregorian,
/// no timezone conversion (the process runs in UTC by convention here).
fn civil_from_days(epoch_secs: u64) -> (i64, u32, u32) {
    let days = (epoch_secs / 86400) as i64;
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mon = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mon <= 2 { y + 1 } else { y };
    (y, mon as u32, d as u32)
}

/// `YYYY-MM-DD HH:MM:SS.mmm` in UTC.
fn timestamp() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let millis = now.subsec_millis();

    let (y, mon, d) = civil_from_days(secs);
    let rem = secs % 86400;
    let h = rem / 3600;
    let min = (rem % 3600) / 60;
    let s = rem % 60;

    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}.{:03}",
        y, mon, d, h, min, s, millis
    )
}

fn log(level: LogLevel, message: &str) {
    let should_log = {
        let state_guard = LOG_STATE.lock();
        state_guard.as_ref().map_or(true, |s| level >= s.level)
    };

    if should_log {
        emit(level, message);
    }
}

/// Log a message at INFO level regardless of the configured level, mirroring
/// the TS `logger.important` used for request logging.
pub fn important(message: &str) {
    emit(LogLevel::Info, message);
}

/// Emit a log line in the TS format: `[LEVEL] YYYY-MM-DD HH:MM:SS.mmm message`.
/// Console shows `[LEVEL]` colored (matching TS `${color}[LEVEL]${reset} ...`).
///
/// Formatting happens outside the lock; the lock only serializes the enqueue,
/// which gives all lines one global order (the writer thread drains FIFO).
/// Before `init()` there is no channel yet — fall back to direct stdout.
fn emit(level: LogLevel, message: &str) {
    let ts = timestamp();
    let plain = format!("[{}] {} {}", level.label(), ts, message);

    let label = format!("[{}]", level.label());
    let colored = Some(format!(
        "{}{}{} {} {}",
        level.color(),
        label,
        "\x1b[0m",
        ts,
        message
    ));

    if let Some(tx) = LOG_SENDER.get() {
        let _ = tx.try_send(LogLine {
            plain,
            colored,
            flush: false,
            quit: false,
        });
    } else if let Some(colored) = colored {
        println!("{}", colored);
    } else {
        println!("{}", plain);
    }
}

pub fn error(message: &str) {
    log(LogLevel::Error, message);
}
pub fn warn(message: &str) {
    log(LogLevel::Warn, message);
}

fn days_from_civil(mut y: i64, m: u32, d: u32) -> i64 {
    if m <= 2 {
        y -= 1;
    }
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u32;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + (doe as i64) - 719468
}

fn parse_log_date(file_name: &str) -> Option<(i64, u32, u32)> {
    let stem = file_name.strip_prefix("mtran-")?.strip_suffix(".log")?;
    let mut parts = stem.split('-');
    let y = parts.next()?.parse::<i64>().ok()?;
    let m = parts.next()?.parse::<u32>().ok()?;
    let d = parts.next()?.parse::<u32>().ok()?;
    if parts.next().is_some() || !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    Some((y, m, d))
}

/// Clean log files exceeding retention days.
pub fn clean_expired_logs(log_dir: &std::path::Path, retention_days: usize) {
    if retention_days == 0 {
        return;
    }

    let read_dir = match fs::read_dir(log_dir) {
        Ok(entries) => entries,
        Err(e) => {
            warn(&format!(
                "Failed to read log directory `{}` for cleanup: {}",
                log_dir.display(),
                e
            ));
            return;
        }
    };

    let (today_y, today_m, today_d) = current_date();
    let current_days = days_from_civil(today_y, today_m, today_d);

    for entry in read_dir {
        let entry = match entry {
            Ok(entry) => entry,
            Err(e) => {
                warn(&format!("Failed to read directory entry: {}", e));
                continue;
            }
        };

        let file_type = match entry.file_type() {
            Ok(ft) => ft,
            Err(e) => {
                warn(&format!(
                    "Failed to get file type for `{}`: {}",
                    entry.path().display(),
                    e
                ));
                continue;
            }
        };

        if !file_type.is_file() {
            continue;
        }

        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if let Some((y, m, d)) = parse_log_date(&name_str) {
            let log_days = days_from_civil(y, m, d);
            if current_days.saturating_sub(log_days) > retention_days as i64 {
                let path = entry.path();
                if let Err(e) = fs::remove_file(&path) {
                    warn(&format!(
                        "Failed to delete expired log file `{}`: {}",
                        path.display(),
                        e
                    ));
                }
            }
        }
    }
}

#[macro_export]
macro_rules! log_error {
	($($arg:tt)*) => {
		$crate::logger::error(&format!($($arg)*))
	};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_filename_format() {
        assert_eq!(log_filename_for_date(2026, 9, 9), "mtran-2026-09-09.log");
        assert_eq!(log_filename_for_date(2026, 1, 5), "mtran-2026-01-05.log");
        assert_eq!(log_filename_for_date(2026, 12, 31), "mtran-2026-12-31.log");
    }

    #[test]
    fn test_civil_conversion_roundtrip() {
        for (y, m, d) in [
            (1970, 1, 1),
            (2024, 2, 29),
            (2026, 9, 9),
            (2026, 12, 31),
            (2030, 6, 15),
        ] {
            let days = days_from_civil(y, m, d);
            let secs = (days as u64) * 86400;
            let (y2, m2, d2) = civil_from_days(secs);
            assert_eq!((y, m, d), (y2, m2, d2));
        }
    }

    #[test]
    fn test_parse_log_date() {
        assert_eq!(parse_log_date("mtran-2026-09-09.log"), Some((2026, 9, 9)));
        assert_eq!(parse_log_date("mtran-2026-01-02.log"), Some((2026, 1, 2)));
        assert_eq!(parse_log_date("mtran-2026.log"), None);
        assert_eq!(parse_log_date("mtran-2026-09.log"), None);
        assert_eq!(parse_log_date("other-2026-09-09.log"), None);
        assert_eq!(parse_log_date("mtran-2026-09-09.txt"), None);
    }

    #[test]
    fn test_clean_expired_logs() {
        let temp_dir = std::env::temp_dir().join(format!(
            "mtran_test_logs_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        fs::create_dir_all(&temp_dir).unwrap();

        let (today_y, today_m, today_d) = current_date();
        let today_days = days_from_civil(today_y, today_m, today_d);

        let expired_days = today_days - 10;
        let (ey, em, ed) = civil_from_days((expired_days as u64) * 86400);
        let expired_file_name = format!("mtran-{:04}-{:02}-{:02}.log", ey, em, ed);
        let expired_file_path = temp_dir.join(&expired_file_name);
        fs::write(&expired_file_path, "expired log content").unwrap();

        let valid_days = today_days - 2;
        let (vy, vm, vd) = civil_from_days((valid_days as u64) * 86400);
        let valid_file_name = format!("mtran-{:04}-{:02}-{:02}.log", vy, vm, vd);
        let valid_file_path = temp_dir.join(&valid_file_name);
        fs::write(&valid_file_path, "valid log content").unwrap();

        let today_file_name = format!("mtran-{:04}-{:02}-{:02}.log", today_y, today_m, today_d);
        let today_file_path = temp_dir.join(&today_file_name);
        fs::write(&today_file_path, "today log content").unwrap();

        let unrelated_file_path = temp_dir.join("other-2020-01-01.log");
        fs::write(&unrelated_file_path, "unrelated").unwrap();

        // retention_days == 0 does not delete anything
        clean_expired_logs(&temp_dir, 0);
        assert!(expired_file_path.exists());
        assert!(valid_file_path.exists());
        assert!(today_file_path.exists());
        assert!(unrelated_file_path.exists());

        // retention_days == 7 deletes files older than 7 days
        clean_expired_logs(&temp_dir, 7);
        assert!(!expired_file_path.exists());
        assert!(valid_file_path.exists());
        assert!(today_file_path.exists());
        assert!(unrelated_file_path.exists());

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
