use std::fmt::Write as FmtWrite;
use std::path::Path;

use crate::runner::SuiteResult;

pub fn generate_junit_xml(
    result: &SuiteResult,
    output_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut xml = String::new();

    xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");

    let total = result.results.len();
    let failures = result.total_failed;
    let time_secs = result.duration_ms as f64 / 1000.0;

    let _ = write!(
        xml,
        "<testsuites tests=\"{total}\" failures=\"{failures}\" time=\"{time_secs:.3}\">\n"
    );
    let _ = write!(
        xml,
        "  <testsuite name=\"difftest\" tests=\"{total}\" failures=\"{failures}\" time=\"{time_secs:.3}\">\n"
    );

    for test_result in &result.results {
        let test_time = test_result.duration_ms as f64 / 1000.0;
        let _ = write!(
            xml,
            "    <testcase name=\"{name}\" classname=\"difftest\" time=\"{time:.3}\"",
            name = xml_escape(&test_result.name),
            time = test_time,
        );

        if test_result.passed {
            xml.push_str(" />\n");
        } else {
            xml.push_str(">\n");

            // Build failure message from metric details
            let mut message = String::new();
            let mut detail = String::new();
            for (name, metric) in &test_result.metrics {
                if !metric.passed {
                    let _ = write!(
                        message,
                        "{name}: mean={mean:.4} threshold={threshold:.4}; ",
                        name = name,
                        mean = metric.mean,
                        threshold = metric.threshold,
                    );
                }
                let _ = write!(
                    detail,
                    "{name}: mean={mean:.4} min={min:.4} max={max:.4} threshold={threshold:.4} {status}\n",
                    name = name,
                    mean = metric.mean,
                    min = metric.min,
                    max = metric.max,
                    threshold = metric.threshold,
                    status = if metric.passed { "PASSED" } else { "FAILED" },
                );
            }

            let _ = write!(
                xml,
                "      <failure message=\"{msg}\">{detail}</failure>\n",
                msg = xml_escape(message.trim_end_matches("; ")),
                detail = xml_escape(&detail),
            );
            xml.push_str("    </testcase>\n");
        }
    }

    xml.push_str("  </testsuite>\n");
    xml.push_str("</testsuites>\n");

    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(output_path, xml)?;
    Ok(())
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}
