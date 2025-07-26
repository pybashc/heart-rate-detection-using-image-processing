# Heart Rate Monitor using Eulerian Video Magnification

This project implements a modern heart rate monitor using a webcam and Eulerian Video Magnification (EVM). It uses facial video input to extract subtle color changes in the skin that correspond to the pulse.

---

## Features

- Face detection with `cvzone`
- Gaussian pyramid decomposition
- Bandpass filtering in frequency domain
- Eulerian amplification of color signals
- Real-time FPS and BPM calculation
- UI overlays for detected BPM, FPS, and status
- Live plotting of BPM over time

---

## Demo

https://user-images.githubusercontent.com/demo-link.gif *(Insert a short screen recording here)*

---

## Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the script:

```bash
python heart_rate_monitor.py
```

To run in headless mode (no video output):

```bash
python heart_rate_monitor.py no-gui
```

Press `q` to quit the program.

---

## File Structure

```text
.
├── heart_rate_monitor.py   # Main source file
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── LICENSE                 # Open-source license (MIT)
└── .gitignore              # Git ignored files
```

---

## Configuration

You can customize the video and processing configuration inside `main()`:

```python
video_config = VideoConfig(real_width=640, real_height=480, frame_rate=30)
processing_config = ProcessingConfig(min_frequency=0.8, max_frequency=3.0, amplification_alpha=150)
```

---

## Limitations

- Accuracy may vary under different lighting conditions.
- Requires a stable webcam feed and a clearly visible face.
- Not a certified medical tool. Use only for educational or experimental purposes.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Credits

- [cvzone](https://github.com/cvzone/cvzone) for face detection and visualization utilities
- OpenCV for video processing
- NumPy for signal processing

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## Author

Arya Gupta

---

## Disclaimer

This tool is not a medically approved device. For health monitoring, consult a certified professional.
