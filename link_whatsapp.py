#!/usr/bin/env python3
"""CLI utility to authenticate and link WhatsApp for automated hazard reporting.

Run this script to scan the WhatsApp Web QR code and link your phone:
    python link_whatsapp.py
"""
import sys

from utils.whatsapp_notifier import WhatsAppNotifier


def main():
    print("=" * 60)
    print("      AI POTHOLE DETECTION — WHATSAPP BOT SETUP")
    print("=" * 60)
    print("This utility connects your WhatsApp account so the detection")
    print("system can automatically dispatch road hazard alerts & photos")
    print("to municipal and highway authorities.")
    print("=" * 60 + "\n")

    notifier = WhatsAppNotifier(enabled=True)
    status = notifier.get_status()

    if status.get("ready"):
        user = status.get("user") or "Connected Account"
        print(f"[ALREADY CONNECTED] WhatsApp is currently linked as: +{user}")
        print("Your session is active. Pothole alerts will dispatch from this number.")
        print("\nTo test, you can run: python main.py --whatsapp --authority-phone <number>")
        return

    print("[INFO] Starting WhatsApp Web client...")
    started = notifier.start_service()
    if not started:
        print("[ERROR] Failed to start WhatsApp microservice.")
        sys.exit(1)

    print("[INFO] Waiting for QR code to generate...")
    success = notifier.wait_for_authentication(timeout_seconds=120)

    if success:
        print("\n" + "=" * 60)
        print("🎉 SUCCESS: WhatsApp is now linked and ready!")
        print("Your credentials have been securely stored in .wwebjs_auth/")
        print("You will NOT need to scan the QR code again.")
        print("=" * 60)
    else:
        print("\n[TIMEOUT] QR code was not scanned in time. Run this script again when ready.")


if __name__ == "__main__":
    main()
