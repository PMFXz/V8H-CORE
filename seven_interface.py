"""
Seven/core/seven_interface.py | PMFX TRADING COMPANY
Role: The Unified Gateway (The Front Gate).
[2025-12-23] Updated: Integrated Trend Master (TM) for 5-Khunpon Synergy
"""

import yaml
import os
from typing import Dict, Any, Optional
from tensorflow.keras.models import load_model

from Seven.core.seven_observer import SevenObserver
from Seven.core.seven_auditor import SevenAuditor
from Seven.core.seven_commander import SevenCommander
# นำเข้า Trend Master เพื่อใช้ในการตรวจสอบ Type หรือทำ Error Handling
# Note: SevenInterface ไม่ได้รัน TM เอง แต่รับ Context มาจาก Main
from trend_master.tm_contract import TrendContext


class SevenInterface:
    def __init__(self, config_path: str = "Seven/config/seven_settings.yaml"):
        # --------------------------------------------------
        # 1. Load Settings (Fail-Safe with Encoding)
        # --------------------------------------------------
        self.settings = self._load_config_safe(config_path)

        # --------------------------------------------------
        # 2. Initialize Sub-Modules
        # --------------------------------------------------
        log_file = self.settings.get("logging", {}).get("log_path", "Seven/data/behavior_logs.csv")
        self.observer = SevenObserver(log_path=log_file)
        self.auditor = SevenAuditor(settings=self.settings)

        # --------------------------------------------------
        # 3. Load Deep Brain (.h5)
        # --------------------------------------------------
        self.brain = self._boot_brain()

        # --------------------------------------------------
        # 4. Initialize Commander (The Decider)
        # --------------------------------------------------
        self.commander = SevenCommander(
            auditor=self.auditor,
            brain_model=self.brain,
            settings=self.settings
        )

        mode = "AI-ENABLED" if self.brain else "LOGIC-ONLY"
        print(f"✅ [SEVEN-CORE] Interface initialized | Mode: {mode} | Support: TM-Integrated")

    # ==================================================
    # Internal Utilities
    # ==================================================
    def _load_config_safe(self, path: str) -> Dict[str, Any]:
        """โหลด Config รองรับภาษาไทยและเช็ค Path อัตโนมัติ"""
        search_paths = [path, "config/seven_settings.yaml", "Seven/config/seven_settings.yaml"]
        
        for p in search_paths:
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding='utf-8') as f:
                        return yaml.safe_load(f) or {}
                except Exception as e:
                    print(f"⚠️ [SEVEN] Config Load Error ({p}): {e}")
        
        print(f"⚠️ [SEVEN] Config not found in any path, using default logic.")
        return {}

    def _boot_brain(self):
        """โหลดโมเดล Seven Brain สำหรับประเมินความมั่นใจ"""
        try:
            model_path = self.settings.get("seven_brain", {}).get("model_path", "Seven/models/seven_brain.h5")
            if os.path.exists(model_path):
                return load_model(model_path)
        except Exception:
            pass # เงียบไว้ถ้าโหลดไม่ได้ (Mode Logic-Only)
        return None

    # ==================================================
    # Public API (The Entrance)
    # ==================================================
    def request_clearance(
        self,
        v8h_signals: dict,
        mz_state: dict,
        acc_info: Any,
        dark_pool: dict = None,
        trend_context: Optional[TrendContext] = None  # <--- เพิ่มการรับ Trend Master
    ) -> Dict[str, Any]:
        """
        ประสานงาน 5 ขุนพลเพื่อตัดสินใจเทรด (V8H, MZ, DP, TM, Seven)
        """
        # 1. Capture State ผ่าน Observer (ส่งต่อ trend_context เข้าไปเก็บ Log ด้วย)
        snapshot = self.observer.capture_snapshot(
            v8h_signals=v8h_signals,
            mz_state=mz_state,
            acc_info=acc_info,
            dark_pool=dark_pool,
            trend_context=trend_context
        )

        # 2. ดึงข้อมูลพอร์ตเบื้องต้น
        balance = getattr(acc_info, "balance", 0.0)
        equity = getattr(acc_info, "equity", 0.0)

        # 3. ส่งให้ Commander พิจารณาร่วมกับขุนพลทั้งหมด
        # Note: Commander จะดึงข้อมูล TM จาก snapshot ที่ Observer เตรียมไว้ให้แล้ว
        cmd = self.commander.get_final_command(
            snapshot=snapshot,
            balance=balance,
            equity=equity
        ) or {}

        # 4. คืนค่า decision ที่คลีนและนำไปใช้ง่าย
        return {
            "allow_trade": bool(cmd.get("allow_trade", False)),
            "lot_scale": float(cmd.get("volume_scale", 1.0)), 
            "mode": cmd.get("system_mode", "OBSERVE"),
            "reason": cmd.get("audit_note", "No data"),
            "trend_label": getattr(trend_context, 'phase', 'N/A') if trend_context else 'N/A'
        }

    def post_trade_analysis(self, result_pts: float):
        """บันทึก Feedback Loop หลังจบงาน"""
        self.observer.log_behavior(result_pts)
        if result_pts < 0:
            # เลขาสรุปงาน: ถ้าแพ้ต้องจดไว้เป็นบทเรียนใน Log ค่ะพี่
            print(f"🛡️ [SEVEN] Behavior Logged: Negative outcome ({result_pts} pts)")
