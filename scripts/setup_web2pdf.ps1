# Setup web2pdf pipeline (Windows PowerShell)
# Run: .\scripts\setup_web2pdf.ps1

Write-Host "=== web2pdf setup ===" -ForegroundColor Cyan

Write-Host "`n[1/3] Installing Python dependencies..."
pip install -r requirements-web2pdf.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: pip install failed." -ForegroundColor Red
    exit 1
}

Write-Host "`n[2/3] Installing Chromium for Playwright..."
python -m playwright install chromium
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Playwright chromium install failed." -ForegroundColor Red
    exit 1
}

Write-Host "`n[3/3] Verifying installation..."
python -c "from web2pdf import Config, Crawler; print('web2pdf package OK')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Import check failed." -ForegroundColor Red
    exit 1
}

Write-Host "`n=== Setup complete! ===" -ForegroundColor Green
Write-Host "Usage: python -m web2pdf crawl https://example.com --depth 2"
