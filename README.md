# 🖐️ Hand Gesture OS — Linux Version

### Touchless Linux Desktop Controller Using Hand Gestures, Eye Tracking & Computer Vision

**Hand Gesture OS** is a real-time computer-vision-based desktop controller that allows users to interact with a Linux desktop using **hand gestures, eye movements, and sound-based clap detection**.

The system uses a webcam to detect hand and facial landmarks and converts recognised gestures into Linux desktop actions through **xdotool**.

It provides touchless control for mouse movement, scrolling, zooming, window management, desktop navigation, and application switching.

---

## 🎯 Problem Statement

Traditional desktop interaction relies on physical input devices such as a mouse and keyboard.

Hand Gesture OS explores an alternative interaction method by using **computer vision and gesture recognition** to translate natural hand and eye movements into operating-system commands.

---

## 💡 How It Works

The system combines three input sources:

```text
              ┌──────────────────┐
              │      Webcam      │
              └────────┬─────────┘
                       │
              ┌────────▼─────────┐
              │     OpenCV       │
              │ Video Processing │
              └────────┬─────────┘
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
    ┌──────────────┐      ┌──────────────┐
    │ MediaPipe    │      │ MediaPipe    │
    │ Hands        │      │ Face Mesh    │
    └──────┬───────┘      └──────┬───────┘
           │                     │
           └──────────┬──────────┘
                      ▼
             ┌─────────────────┐
             │ Gesture / Eye   │
             │ Classification  │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │    xdotool      │
             │ Linux OS Control│
             └─────────────────┘

             Microphone
                  │
                  ▼
          SoundDevice + NumPy
                  │
                  ▼
             Clap Detection
                  │
                  ▼
              Alt + Tab
```

---

# ✨ Features

## 🖱️ Touchless Mouse Control

A **pinch gesture** between the thumb and index finger activates pointer mode.

The system detects the index-finger position and maps it to the screen coordinates, allowing the user to control the mouse without touching a physical mouse.

The implementation uses `xdotool mousemove` for pointer movement.

---

## 📜 Gesture-Based Scrolling

Two fingers can be used to activate scrolling mode.

| Gesture                      | Action      |
| ---------------------------- | ----------- |
| ✌️ Two fingers pointing UP   | Scroll Up   |
| ✌️ Two fingers pointing DOWN | Scroll Down |

The application maps these actions to Linux mouse-wheel events through `xdotool`.

Scrolling can be continuously triggered while the user's eyes remain closed.

---

## 🔍 Zoom Control

The system supports gesture-controlled zooming.

| Gesture                | Mode     |
| ---------------------- | -------- |
| ☝️ One finger + thumb  | Zoom In  |
| ✌️ Two fingers + thumb | Zoom Out |

The corresponding Linux keyboard commands are:

```text
Ctrl + +
Ctrl + =
Ctrl + -
```

The zoom action is triggered through an eye-blink event after the gesture mode has been activated.

---

## 🪟 Window Management

Several common Linux desktop operations can be performed using hand gestures.

| Gesture                  | Action                       |
| ------------------------ | ---------------------------- |
| ✊ Closed palm with thumb | Close current window         |
| ☝️ One finger, thumb OFF | Open applications / overview |
| 🖐️ Open palm            | Show desktop                 |

The project maps these gestures to:

```text
Super
Super + D
Alt + F4
```

through `xdotool`.

---

## 👏 Clap-Based Application Switching

The system also supports **clap detection** through the microphone.

A clap triggers:

```text
Alt + Tab
```

The audio stream is processed using SoundDevice, and the project calculates the RMS volume of the incoming audio. A clap is detected when the RMS level crosses the configured threshold while respecting a minimum interval between detections.

Default settings include:

```text
Sample Rate:       16000 Hz
Block Duration:    0.1 seconds
Clap Threshold:    0.35
Minimum Interval:  0.4 seconds
```

---

# 🖐️ Gesture Reference

| Gesture / Input          | Result                                 |
| ------------------------ | -------------------------------------- |
| 🤏 Thumb + index pinch   | Mouse pointer mode                     |
| ✌️ Two fingers UP        | Scroll Up                              |
| ✌️ Two fingers DOWN      | Scroll Down                            |
| ☝️ One finger + thumb    | Zoom In                                |
| ✌️ Two fingers + thumb   | Zoom Out                               |
| ✊ Closed palm + thumb    | Close Window                           |
| ☝️ One finger, thumb OFF | Show Applications                      |
| 🖐️ Open palm            | Show Desktop                           |
| 👁️ Closed eyes          | Trigger continuous scroll / navigation |
| 👁️ Blink                | Trigger zoom                           |
| 👏 Clap                  | Alt + Tab                              |

---

# 🧠 Gesture Recognition

The project uses **MediaPipe Hands** to detect hand landmarks.

MediaPipe provides hand landmark coordinates that are used to determine:

* Extended fingers
* Thumb state
* Hand orientation
* Pinch distance
* Two-finger direction
* Hand gesture type

The implementation recognises gestures including:

```text
PINCH
OPEN_PALM
CLOSED_PALM
FIST
TWO_FINGERS
UNKNOWN
```

For two-finger gestures, the system also determines the direction:

```text
UP
DOWN
LEFT
RIGHT
```

---

# 👁️ Eye Tracking

MediaPipe Face Mesh is used for facial landmark detection.

The application analyses eye states to identify:

* Eyes open
* Eyes closed
* Blink events
* Eye-based interaction states

The current implementation supports multiple detected faces and uses the first detected face for the primary eye-control state.

---

# ⏱️ Gesture Hold System

To reduce accidental commands, gesture modes are not activated immediately.

The application requires the gesture to be held for:

```text
1.5 seconds
```

Continuous actions use a repeat interval of:

```text
0.15 seconds
```

These values are configurable in `preset.py`.

---

# 🏗️ System Architecture

```text
                 INPUT LAYER
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
     Webcam       Webcam       Microphone
        │            │            │
        ▼            ▼            ▼
   Hand Tracking  Face Mesh   Audio Stream
        │            │            │
        └──────┬─────┘            │
               │                  │
               ▼                  ▼
        Gesture / Eye       RMS Clap Detection
         Classification            │
               │                  │
               └────────┬─────────┘
                        ▼
                 Command Mapping
                        │
                        ▼
                    xdotool
                        │
                        ▼
                 Linux Desktop
```

---

# 🛠️ Technology Stack

| Technology              | Purpose                             |
| ----------------------- | ----------------------------------- |
| **Python**              | Core programming language           |
| **OpenCV**              | Webcam capture and image processing |
| **MediaPipe Hands**     | Hand landmark detection             |
| **MediaPipe Face Mesh** | Eye and facial landmark tracking    |
| **SoundDevice**         | Microphone/audio input              |
| **NumPy**               | Audio RMS calculation               |
| **xdotool**             | Linux mouse and keyboard automation |
| **subprocess**          | Executing Linux system commands     |
| **math / time**         | Gesture calculations and timing     |

The main implementation imports OpenCV, MediaPipe, SoundDevice, NumPy, `subprocess`, `math`, and `time`, while Linux desktop control is performed through `xdotool`.

---

# 📂 Project Structure

```text
Hand-Gesture-OS-demo---Linux-version/
│
├── preset.py
│
├── hand_gesture_demo.py.save
│
├── External Parts/
│
├── Tests/
│
├── .gitignore
│
└── README.md
```

The current repository contains the main `preset.py` implementation along with the `External Parts` and `Tests` directories and the saved gesture-demo file.

---

# ⚙️ Requirements

This project is designed for a **Linux desktop environment**.

You need:

* Linux operating system
* Python 3.x
* Working webcam
* Microphone
* `xdotool`
* Python packages used by `preset.py`

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/sandip2810-bhp/Hand-Gesture-OS-demo---Linux-version.git
```

Move into the project:

```bash
cd Hand-Gesture-OS-demo---Linux-version
```

---

## 2. Install xdotool

`xdotool` is required because it performs the actual Linux mouse and keyboard operations.

On Ubuntu/Debian-based systems:

```bash
sudo apt update
sudo apt install xdotool
```

The application itself reports this installation command if `xdotool` is not found.

---

## 3. Create a Virtual Environment

Recommended:

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

---

## 4. Install Python Dependencies

Install the packages used by the main program:

```bash
pip install opencv-python mediapipe sounddevice numpy
```

---

# ▶️ Run the Application

Start the main program:

```bash
python3 preset.py
```

The application starts webcam processing and the audio input stream.

You should see:

```text
🎧 Clap Alt+Tab active: every clap = Alt+Tab
🎥 Hand / eye controller running. Press 'q' to quit.
```

The application uses a webcam resolution of approximately:

```text
960 × 720
```

and opens the camera using OpenCV.

Press:

```text
Q
```

to exit the application.

---

# 🎥 Camera Configuration

The current implementation initializes OpenCV with:

```text
Camera: 0
Width: 960
Height: 720
```

MediaPipe Hands is configured for up to **4 hands**, while Face Mesh is configured for up to **3 faces**.

---

# 🔊 Audio Configuration

The clap detector uses:

```text
Sample Rate: 16000
Block Duration: 0.1 seconds
Threshold: 0.35
Minimum Clap Interval: 0.4 seconds
```

These values can be adjusted in `preset.py` depending on the microphone and surrounding environment.

---

# ⚠️ Important Notes

### Linux Only

This version specifically uses **Linux desktop commands through `xdotool`**.

For example:

```text
Super
Super + D
Alt + F4
Ctrl + +
Ctrl + -
Alt + Tab
Mouse movement
Mouse-wheel events
```

### Webcam Required

Hand and eye recognition depend on a working webcam.

### Microphone Required

The clap-based `Alt + Tab` functionality requires microphone access.

### Lighting Matters

Hand and facial landmark detection can be affected by:

* Poor lighting
* Camera quality
* Hand occlusion
* Distance from camera
* Background conditions

### Gesture Stability

Most gesture commands require a **1.5-second hold** before activation, helping reduce accidental triggers.

---

# 🧪 Debugging

If the application does not control the desktop:

### Check xdotool

```bash
xdotool --version
```

If it is missing:

```bash
sudo apt install xdotool
```

### Check Webcam

Make sure the webcam is accessible and not being used by another application.

### Check Microphone

Make sure Linux has permission to access the microphone and that the correct input device is selected.

---

# 🔮 Future Improvements

Potential improvements include:

* Custom gesture mapping
* Gesture calibration
* Better false-positive filtering
* Improved hand tracking stability
* User-specific sensitivity settings
* More Linux desktop actions
* Custom application controls
* Voice commands
* Multi-monitor support
* Better audio-based clap detection
* Wayland compatibility improvements
* Cross-platform support

---

# 🧠 Learning Outcomes

This project demonstrates practical experience with:

* Computer Vision
* Real-time video processing
* MediaPipe
* Hand landmark detection
* Facial landmark detection
* Eye tracking
* Gesture classification
* Audio signal processing
* Linux desktop automation
* Shell/system command integration
* Real-time event handling
* Human-computer interaction

---

# 👨‍💻 Author

**Sandip Mandal**

Computer Science Engineering Student & Developer

---

# ⭐ Project

If you find this project interesting, consider giving the repository a ⭐ on GitHub.

---

## 🧰 Core Technologies

**Python • OpenCV • MediaPipe • SoundDevice • NumPy • xdotool • Computer Vision • Gesture Recognition • Eye Tracking • Linux Automation**
