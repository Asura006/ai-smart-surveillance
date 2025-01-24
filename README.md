
Overview

This project implements a face recognition-based attendance system using OpenCV and Python. It detects faces in real-time from a webcam feed, matches them against a database of known faces, and retrieves their corresponding status (e.g., Hosteller or Day Scholar). The system sends email notifications for each non hosteler or unknown person detected individual.

Features

Real-time face detection using OpenCV's Haar cascades.

Face recognition using template matching (Local Binary Patterns Histogram - LBPH can also be used).

Metadata integration to associate each face with a name and status.

Logs attendance with timestamps and camera details.

Sends email notifications for each non-hosteler detection.
