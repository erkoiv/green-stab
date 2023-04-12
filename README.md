# Intro

This demo shows capability of using machine vision to remove sway from a static object on a greenscreen background.

The rotation of the body cannot be removed but there is major sway reduction.

You can use your own webcam stream to test this by setting the video_in variable to 0 in the code to designate webcam stream 0 as the input.

A blue marker has been added to the expected center of attention (face) when looking at the video to enunciate effect of sway reduction. This is not used in the stabilization.

# Video used for demo

[young man walking on a green screen chroma key UXYGUUT](https://www.youtube.com/watch?v=tvdEO4dlgJY)

# Face detection

Uses model that comes with CV2. It has been included.

# Demo

Unstabilized (left) vs. stabilized (right)

![Unstabilized on the left and stabilized on the right](walkingStab.gif)
