# Browser presentation demo

Live demo: https://zhandolia.github.io/Muscle-Intelligence/

This static browser adaptation presents the existing mobile prototype's push-up workflow without requiring an account, a mobile installation, or a running backend. The React Native application remains unchanged.

## Presenting

1. Select **First attempt**, play the original clip, and choose **View skeleton**.
2. Choose **View sample feedback** to reveal the prerecorded colored overlay and preset feedback.
3. Open **View reference example**, or use **Compare attempts** and **Play both clips** to show the change between samples.
4. Use **Reset demo** to restart the presentation.

The demo works on desktop and mobile. **Preview your own video** uses a local browser object URL: it does not upload the file or analyze it. Sample overlays and labels are explicitly marked as prerecorded. There is no camera capture, authentication, cloud storage, live inference, or injury-risk assessment in this demo.

## Source mapping

- First attempt: `screens/upload/UploadBetta.js`, `SkeletonBetta.js`, and `FinalBetta.js`.
- Second attempt: `screens/upload/UploadAlpha.js`, `SkeletonAlpha.js`, and `FinalAlpha.js`.
- Reference example: `screens/upload/StasSample.js`.
- Video assets: H.264 conversions of the corresponding GIFs in the repository's `assets/` directory, preserving the original frames and timing. Poster images are first-frame extracts.
- “Needs attention” is the demo's presentation label for the first sample's shoulders warning; no diagnostic claim is made.

## Running and deploying

No dependencies or build step are required. From the repository root:

```sh
python -m http.server 8080 --directory docs
```

GitHub Pages publishes the `docs/` directory from `main`. Commit changes under this directory and push to redeploy. All application files and videos use relative paths so the project URL works correctly. Only this directory is published; mobile configuration, environment files, and authentication dependencies are not included.

Validation covers both sample workflows, comparison playback, reference playback, local video preview, reset, mobile layout, static assets, and JavaScript syntax. Test any new video in a browser after converting it.
