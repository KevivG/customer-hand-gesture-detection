from setuptools import setup

APP=['Triathlon-customer_hand_gesture_detection.py']
OPTIONS = {
    'argv_emulation':True,
}

setup(
    app=APP,
    options={'py2app': OPTIONS},
    setup_require=['py2app']
)
