import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


VEO_URL_RE = re.compile(r"^https?://(?:app\.)?veo\.co/", re.IGNORECASE)
VEO_LOGIN_URL = "https://app.veo.co/login"


def is_veo_url(text: str) -> bool:
    return bool(VEO_URL_RE.match(text.strip()))


@dataclass
class VeoSession:
    """Module-level singleton holding a Playwright storage_state (cookies + localStorage)
    for an authenticated Veo session. One login per Colab session."""

    storage_state: Optional[dict] = None
    email: Optional[str] = None
    connected: bool = False
    last_error: Optional[str] = None

    def login(self, email: str, password: str, timeout_ms: int = 30000) -> Tuple[bool, str]:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return False, "playwright n'est pas installé — `pip install playwright && playwright install chromium`"

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
                context = browser.new_context()
                page = context.new_page()
                page.goto(VEO_LOGIN_URL, wait_until="domcontentloaded", timeout=timeout_ms)

                self._dismiss_cookies(page)
                self._fill_login(page, email, password)
                self._submit_login(page)

                try:
                    page.wait_for_url(
                        lambda url: "/login" not in url and "auth" not in url.lower(),
                        timeout=timeout_ms,
                    )
                except Exception:
                    err = self._extract_login_error(page)
                    browser.close()
                    self.last_error = err
                    return False, err

                self.storage_state = context.storage_state()
                self.email = email
                self.connected = True
                self.last_error = None
                browser.close()
                return True, f"✅ Connecté en tant que {email}"
        except Exception as e:
            self.last_error = str(e)
            return False, f"❌ Login Veo échoué : {type(e).__name__} — {e}"

    @staticmethod
    def _dismiss_cookies(page) -> None:
        for sel in [
            'button:has-text("Accept")', 'button:has-text("Accepter")',
            'button:has-text("I agree")', 'button[id*="accept"]',
            '[aria-label*="accept" i]',
        ]:
            try:
                page.locator(sel).first.click(timeout=1500)
                return
            except Exception:
                pass

    @staticmethod
    def _fill_login(page, email: str, password: str) -> None:
        email_sel_candidates = [
            'input[type="email"]',
            'input[name="email"]',
            'input[name="username"]',
            'input[autocomplete="email"]',
            'input[id*="email" i]',
        ]
        for sel in email_sel_candidates:
            try:
                page.locator(sel).first.fill(email, timeout=3000)
                break
            except Exception:
                continue

        pw_sel_candidates = [
            'input[type="password"]',
            'input[name="password"]',
            'input[autocomplete="current-password"]',
        ]
        for sel in pw_sel_candidates:
            try:
                page.locator(sel).first.fill(password, timeout=3000)
                break
            except Exception:
                continue

    @staticmethod
    def _submit_login(page) -> None:
        for sel in [
            'button[type="submit"]',
            'button:has-text("Log in")',
            'button:has-text("Sign in")',
            'button:has-text("Connexion")',
            'button:has-text("Continue")',
        ]:
            try:
                page.locator(sel).first.click(timeout=3000)
                return
            except Exception:
                continue
        page.keyboard.press("Enter")

    @staticmethod
    def _extract_login_error(page) -> str:
        for sel in ['[role="alert"]', '.error', '[class*="error" i]']:
            try:
                txt = page.locator(sel).first.inner_text(timeout=1000)
                if txt and len(txt) < 400:
                    return f"❌ Veo a refusé : {txt.strip()}"
            except Exception:
                pass
        url = page.url
        if "/login" in url or "auth" in url.lower():
            return "❌ Toujours sur la page login (creds invalides ? captcha ? 2FA ?)"
        return f"❌ Login indéterminé (URL actuelle : {url})"

    def extract_mp4(self, match_url: str, output_path: Path, m3u8_timeout_ms: int = 45000) -> Path:
        if not self.connected or self.storage_state is None:
            raise RuntimeError("Veo : connecte-toi d'abord (panneau 🔐 Veo)")

        from playwright.sync_api import sync_playwright

        captured: dict = {}

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            context = browser.new_context(storage_state=self.storage_state)
            page = context.new_page()

            def on_request(req):
                if captured:
                    return
                if ".m3u8" in req.url and "veo" in req.url:
                    captured["url"] = req.url
                    captured["headers"] = dict(req.headers)

            page.on("request", on_request)
            page.goto(match_url, wait_until="domcontentloaded", timeout=m3u8_timeout_ms)
            self._dismiss_cookies(page)

            for sel in [
                'button[aria-label*="play" i]',
                'button:has-text("Play")',
                'video',
                '[class*="play-button" i]',
            ]:
                try:
                    page.locator(sel).first.click(timeout=4000)
                    break
                except Exception:
                    continue

            elapsed = 0
            poll_ms = 500
            while not captured and elapsed < m3u8_timeout_ms:
                page.wait_for_timeout(poll_ms)
                elapsed += poll_ms

            if not captured:
                browser.close()
                raise RuntimeError("Pas de stream .m3u8 capturé — Veo a peut-être changé son player, ou tu n'as pas accès à ce match")

            cookies = context.cookies(captured["url"])
            cookies_header = "; ".join(f"{c['name']}={c['value']}" for c in cookies)
            referer = match_url
            user_agent = captured["headers"].get("user-agent", "")
            browser.close()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        headers_blob = (
            f"Cookie: {cookies_header}\r\n"
            f"Referer: {referer}\r\n"
            + (f"User-Agent: {user_agent}\r\n" if user_agent else "")
        )

        cmd = [
            "ffmpeg", "-y",
            "-headers", headers_blob,
            "-i", captured["url"],
            "-c", "copy",
            "-bsf:a", "aac_adtstoasc",
            str(output_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not output_path.exists():
            tail = (proc.stderr or "")[-800:]
            raise RuntimeError(f"ffmpeg HLS download failed:\n{tail}")
        return output_path


_session = VeoSession()


def get_session() -> VeoSession:
    return _session
