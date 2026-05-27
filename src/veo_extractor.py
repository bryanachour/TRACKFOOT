import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


VEO_URL_RE = re.compile(r"^https?://(?:app\.)?veo\.co/", re.IGNORECASE)
VEO_ROOT_URL = "https://app.veo.co/"
VEO_LOGIN_CANDIDATES = [
    "https://app.veo.co/",
    "https://app.veo.co/login",
    "https://app.veo.co/login/",
    "https://app.veo.co/sign-in",
    "https://app.veo.co/signin",
]


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
                browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
                context = browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
                    )
                )
                page = context.new_page()

                reached_form = self._navigate_to_login_form(page, timeout_ms)
                if not reached_form:
                    body = self._safe_text(page, "body")[:400]
                    browser.close()
                    err = f"❌ Impossible de trouver le form de login (page actuelle : {page.url}). Extrait : {body}"
                    self.last_error = err
                    return False, err

                self._dismiss_cookies(page)
                self._fill_login(page, email, password)
                self._submit_login(page)

                logged_in = self._wait_for_logged_in(page, timeout_ms)
                if not logged_in:
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

    def _navigate_to_login_form(self, page, timeout_ms: int) -> bool:
        for url in VEO_LOGIN_CANDIDATES:
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                page.wait_for_load_state("networkidle", timeout=8000)
            except Exception:
                continue

            if self._has_login_form(page):
                return True

            for sel in [
                'a:has-text("Log in")', 'a:has-text("Login")',
                'a:has-text("Sign in")', 'a:has-text("Connexion")',
                'button:has-text("Log in")', 'button:has-text("Login")',
            ]:
                try:
                    page.locator(sel).first.click(timeout=2000)
                    page.wait_for_load_state("networkidle", timeout=8000)
                    if self._has_login_form(page):
                        return True
                except Exception:
                    continue
        return self._has_login_form(page)

    @staticmethod
    def _has_login_form(page) -> bool:
        for sel in [
            'input[type="email"]', 'input[name="email"]',
            'input[name="username"]', 'input[autocomplete="email"]',
        ]:
            try:
                if page.locator(sel).count() > 0:
                    return True
            except Exception:
                pass
        return False

    @staticmethod
    def _wait_for_logged_in(page, timeout_ms: int) -> bool:
        deadline = timeout_ms
        try:
            page.wait_for_url(
                lambda url: "login" not in url.lower() and "sign-in" not in url.lower() and "auth" not in url.lower(),
                timeout=deadline,
            )
            return True
        except Exception:
            pass
        try:
            page.wait_for_load_state("networkidle", timeout=8000)
        except Exception:
            pass
        return "login" not in page.url.lower() and "sign-in" not in page.url.lower()

    @staticmethod
    def _safe_text(page, selector: str) -> str:
        try:
            return page.locator(selector).first.inner_text(timeout=1000) or ""
        except Exception:
            return ""

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

    def extract_mp4(self, match_url: str, output_path: Path, m3u8_timeout_ms: int = 60000) -> Path:
        if not self.connected or self.storage_state is None:
            raise RuntimeError("Veo : connecte-toi d'abord (panneau 🔐 Veo)")

        from playwright.sync_api import sync_playwright

        captured: dict = {}
        media_urls: List[str] = []
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
            context = browser.new_context(
                storage_state=self.storage_state,
                user_agent=(
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
                ),
            )
            page = context.new_page()

            def on_request(req):
                url = req.url
                lo = url.lower()
                if any(tok in lo for tok in [".m3u8", ".mpd", ".mp4", ".m4s", ".ts", "playlist", "manifest"]):
                    media_urls.append(url)
                if captured:
                    return
                if ".m3u8" in lo or ".mpd" in lo:
                    captured["url"] = url
                    captured["headers"] = dict(req.headers)
                    captured["kind"] = "hls" if ".m3u8" in lo else "dash"

            page.on("request", on_request)
            page.goto(match_url, wait_until="domcontentloaded", timeout=m3u8_timeout_ms)
            try:
                page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                pass
            self._dismiss_cookies(page)

            for sel in [
                'video',
                'button[aria-label*="play" i]',
                'button[title*="play" i]',
                'button:has-text("Play")',
                '[class*="play-button" i]',
                '[class*="PlayButton" i]',
                '[data-testid*="play" i]',
            ]:
                try:
                    page.locator(sel).first.click(timeout=3000)
                    break
                except Exception:
                    continue
            try:
                page.evaluate("document.querySelectorAll('video').forEach(v => { v.muted = true; v.play().catch(() => {}); })")
            except Exception:
                pass

            elapsed = 0
            poll_ms = 500
            while not captured and elapsed < m3u8_timeout_ms:
                page.wait_for_timeout(poll_ms)
                elapsed += poll_ms

            if not captured:
                debug_dir = output_path.parent
                debug_dir.mkdir(parents=True, exist_ok=True)
                stamp = int(time.time())
                shot_path = debug_dir / f"veo_debug_{stamp}.png"
                try:
                    page.screenshot(path=str(shot_path), full_page=False)
                except Exception:
                    shot_path = None
                title = ""
                try:
                    title = page.title()
                except Exception:
                    pass
                url_now = page.url
                seen = "\n  ".join(dict.fromkeys(media_urls[:15])) or "(rien)"
                browser.close()
                raise RuntimeError(
                    "Pas de stream HLS/DASH détecté dans la page.\n"
                    f"URL actuelle : {url_now}\n"
                    f"Titre : {title}\n"
                    f"URLs media vues :\n  {seen}\n"
                    + (f"Screenshot debug : {shot_path}" if shot_path else "")
                )

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
