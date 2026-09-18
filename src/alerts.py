"""
Push alert integration for the Energy Anomaly Detector.

Built channel: Email (smtplib — no extra dependencies)

Supported channels (not yet implemented):
  - SMS  : Twilio API  (pip install twilio)
  - Slack: Incoming Webhooks  (pip install slack-sdk)
  - Push : ntfy.sh self-hosted  (plain HTTP POST, no account needed)

Setup (Email / Gmail)
---------------------
1. Enable 2-Step Verification on your Google account.
2. Google Account → Security → App Passwords → generate one for "Mail".
3. Set two environment variables (never hardcode credentials):

   Windows PowerShell:
       $env:ALERT_EMAIL          = "you@gmail.com"
       $env:ALERT_EMAIL_PASSWORD = "xxxx xxxx xxxx xxxx"   # 16-char app password

   Linux / macOS:
       export ALERT_EMAIL="you@gmail.com"
       export ALERT_EMAIL_PASSWORD="xxxx xxxx xxxx xxxx"

   Or put them in a .env file and load with python-dotenv (optional).

4. Set ALERT_RECIPIENT if you want alerts sent to a different address;
   otherwise alerts go back to the sender.
"""

import os
import smtplib
import warnings
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Walk up from src/ to find the .env at the project root
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass  # dotenv optional — env vars can also be set directly in the shell


def send_email_alert(
    anomaly_time,
    usage_value: float,
    expected_value: float,
    anomaly_score: float,
    *,
    smtp_host: str = "smtp.gmail.com",
    smtp_port: int = 465,
) -> bool:
    """
    Fire an email alert for a flagged anomaly.

    Parameters
    ----------
    anomaly_time : datetime-like
        Timestamp of the anomalous hour.
    usage_value : float
        Observed hourly average power consumption (kW).
    expected_value : float
        24-h rolling mean baseline (kW) — what "normal" looks like.
    anomaly_score : float
        Raw Isolation Forest decision score (lower = more anomalous).
    smtp_host : str
        SMTP server hostname.  Default: smtp.gmail.com
    smtp_port : int
        SMTP SSL port.  Default: 465

    Returns
    -------
    bool
        True if the email was sent successfully, False otherwise.
        Failures are warned (not raised) so the pipeline continues.

    Environment variables required
    ------------------------------
    ALERT_EMAIL          — sender address (also used as recipient unless
                           ALERT_RECIPIENT is set)
    ALERT_EMAIL_PASSWORD — app password for the sender account
    """
    sender = os.environ.get("ALERT_EMAIL")
    password = os.environ.get("ALERT_EMAIL_PASSWORD")
    recipient = os.environ.get("ALERT_RECIPIENT", sender)

    if not sender or not password:
        warnings.warn(
            "Email alert skipped: ALERT_EMAIL or ALERT_EMAIL_PASSWORD "
            "environment variable is not set.",
            stacklevel=2,
        )
        return False

    ratio = usage_value / expected_value if expected_value > 0 else float("inf")
    ts_str = str(anomaly_time)

    subject = f"[Energy Alert] Anomaly at {ts_str} — {usage_value:.2f} kW ({ratio:.1f}x normal)"

    plain_body = (
        f"Anomaly detected at {ts_str}.\n\n"
        f"  Observed usage : {usage_value:.2f} kW\n"
        f"  Expected (24h baseline) : {expected_value:.2f} kW\n"
        f"  Ratio vs baseline : {ratio:.1f}x\n"
        f"  Anomaly score : {anomaly_score:.4f}  (lower = more anomalous)\n\n"
        f"Check your appliances and devices for anything left on unexpectedly.\n\n"
        f"---\n"
        f"Sent by AI-Based Energy Anomaly Detector\n"
        f"Powered by Isolation Forest (scikit-learn)\n"
    )

    html_body = f"""\
<html><body style="font-family:sans-serif;color:#1e293b">
  <h2 style="color:#dc2626">&#9888; Energy Usage Anomaly Detected</h2>
  <p>An anomaly was flagged at <strong>{ts_str}</strong>.</p>
  <table cellpadding="6" style="border-collapse:collapse;width:380px">
    <tr style="background:#fef2f2">
      <td><strong>Observed usage</strong></td>
      <td>{usage_value:.2f} kW</td>
    </tr>
    <tr>
      <td><strong>Expected (24h baseline)</strong></td>
      <td>{expected_value:.2f} kW</td>
    </tr>
    <tr style="background:#fef2f2">
      <td><strong>Ratio vs baseline</strong></td>
      <td>{ratio:.1f}&times; normal</td>
    </tr>
    <tr>
      <td><strong>Anomaly score</strong></td>
      <td>{anomaly_score:.4f} &nbsp;<em>(lower = more anomalous)</em></td>
    </tr>
  </table>
  <p style="margin-top:16px">Check your appliances and devices for anything left on unexpectedly.</p>
  <hr style="border:none;border-top:1px solid #e5e7eb"/>
  <p style="font-size:12px;color:#94a3b8">
    Sent by AI-Based Energy Anomaly Detector &mdash; Isolation Forest (scikit-learn)
  </p>
</body></html>"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    msg.attach(MIMEText(plain_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
            server.login(sender, password)
            server.sendmail(sender, recipient, msg.as_string())
        print(f"  [alert] Email sent to {recipient} for anomaly at {ts_str}")
        return True
    except smtplib.SMTPAuthenticationError:
        warnings.warn(
            "Email alert failed: authentication error. "
            "Check ALERT_EMAIL / ALERT_EMAIL_PASSWORD and ensure the App Password is correct.",
            stacklevel=2,
        )
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Email alert failed: {exc}", stacklevel=2)
    return False
