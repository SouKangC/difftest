use std::fmt::Write as FmtWrite;
use std::path::Path;

use crate::error;
use crate::runner::SuiteResult;

pub fn generate_html_report(
    result: &SuiteResult,
    output_path: &Path,
) -> error::Result<()> {
    let mut html = String::new();

    write_header(&mut html);
    write_summary(&mut html, result);

    for test_result in &result.results {
        write_test_section(&mut html, test_result);
    }

    write_footer(&mut html);

    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(output_path, html)?;
    Ok(())
}

fn write_header(html: &mut String) {
    html.push_str(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>difftest Report</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 2rem; background: #f8f9fa; color: #212529; }
  h1 { border-bottom: 2px solid #dee2e6; padding-bottom: 0.5rem; }
  .summary { display: flex; gap: 2rem; margin-bottom: 2rem; }
  .summary-card { background: white; padding: 1rem 1.5rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  .passed { color: #198754; }
  .failed { color: #dc3545; }
  .test-section { background: white; padding: 1.5rem; border-radius: 8px; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  .test-header { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem; }
  .badge { padding: 0.25rem 0.75rem; border-radius: 4px; font-size: 0.85rem; font-weight: 600; color: white; }
  .badge-pass { background: #198754; }
  .badge-fail { background: #dc3545; }
  table { width: 100%; border-collapse: collapse; margin-top: 0.75rem; }
  th, td { text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid #dee2e6; }
  th { background: #f8f9fa; font-weight: 600; }
  .image-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 1rem; margin-top: 1rem; }
  .image-card { text-align: center; }
  .image-card img { max-width: 100%; border-radius: 4px; border: 1px solid #dee2e6; }
  .image-card p { margin: 0.25rem 0; font-size: 0.85rem; color: #6c757d; }
  .comparison { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem; }
  .comparison-label { font-weight: 600; text-align: center; margin-bottom: 0.5rem; }
  footer { margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #dee2e6; font-size: 0.85rem; color: #6c757d; }
</style>
</head>
<body>
<h1>difftest Report</h1>
"#,
    );
}

fn write_summary(html: &mut String, result: &SuiteResult) {
    let _ = write!(
        html,
        r#"<div class="summary">
  <div class="summary-card"><strong>Total Tests:</strong> {}</div>
  <div class="summary-card"><strong class="passed">Passed:</strong> {}</div>
  <div class="summary-card"><strong class="failed">Failed:</strong> {}</div>
  <div class="summary-card"><strong>Duration:</strong> {}ms</div>
</div>
"#,
        result.results.len(),
        result.total_passed,
        result.total_failed,
        result.duration_ms,
    );
}

fn write_test_section(html: &mut String, test_result: &crate::runner::TestResult) {
    let (badge_class, status_text) = if test_result.passed {
        ("badge-pass", "PASSED")
    } else {
        ("badge-fail", "FAILED")
    };

    let _ = write!(
        html,
        r#"<div class="test-section">
  <div class="test-header">
    <span class="badge {badge_class}">{status_text}</span>
    <h2 style="margin:0">{name}</h2>
    <span style="color:#6c757d; font-size:0.85rem">({duration}ms)</span>
  </div>
"#,
        badge_class = badge_class,
        status_text = status_text,
        name = html_escape(&test_result.name),
        duration = test_result.duration_ms,
    );

    // Metric table
    if !test_result.metrics.is_empty() {
        html.push_str(
            r#"  <table>
    <tr><th>Metric</th><th>Mean</th><th>Min</th><th>Max</th><th>Threshold</th><th>Status</th></tr>
"#,
        );

        for (name, metric) in &test_result.metrics {
            let status = if metric.passed {
                r#"<span class="passed">&#10003;</span>"#
            } else {
                r#"<span class="failed">&#10007;</span>"#
            };
            let _ = write!(
                html,
                "    <tr><td>{}</td><td>{:.4}</td><td>{:.4}</td><td>{:.4}</td><td>{:.4}</td><td>{}</td></tr>\n",
                html_escape(name),
                metric.mean,
                metric.min,
                metric.max,
                metric.threshold,
                status,
            );
        }
        html.push_str("  </table>\n");
    }

    // Image grid
    if !test_result.images.is_empty() {
        html.push_str("  <div class=\"image-grid\">\n");
        for img in &test_result.images {
            let _ = write!(
                html,
                r#"    <div class="image-card">
      <img src="{path}" alt="{prompt}">
      <p>{prompt}</p>
      <p>seed: {seed}</p>
    </div>
"#,
                path = html_escape(&img.path),
                prompt = html_escape(&img.prompt),
                seed = img.seed,
            );
        }
        html.push_str("  </div>\n");
    }

    html.push_str("</div>\n");
}

fn write_footer(html: &mut String) {
    html.push_str(
        r#"<footer>Generated by <strong>difftest</strong></footer>
</body>
</html>
"#,
    );
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
