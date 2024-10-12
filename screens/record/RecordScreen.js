import React, { useState, useEffect, useRef } from 'react';
import { View, Text, StyleSheet, Button } from 'react-native';
import { Camera } from 'expo-camera'; // Ensure this import is present
import { Video } from 'expo-av';

export default function RecordScreen() {
  const [hasCameraPermission, setHasCameraPermission] = useState(null);
  const [hasAudioPermission, setHasAudioPermission] = useState(null);
  const [camera, setCamera] = useState(null);
  const [recording, setRecording] = useState(null);
  
  // Add this check to prevent accessing undefined properties
  const [type, setType] = useState(Camera ? Camera.Constants.Type.back : null); 

  const [flashMode, setFlashMode] = useState(Camera.Constants.FlashMode.off);
  const [autoFocus, setAutoFocus] = useState(Camera.Constants.AutoFocus.on);
  const video = useRef(null);
  const [status, setStatus] = useState({});

  // Request camera and audio permissions
  useEffect(() => {
    (async () => {
      const cameraStatus = await Camera.requestPermissionsAsync();
      setHasCameraPermission(cameraStatus.status === 'granted');

      const audioStatus = await Camera.requestMicrophonePermissionsAsync();
      setHasAudioPermission(audioStatus.status === 'granted');
    })();
  }, []);

  // Start video recording
  const takeVideo = async () => {
    if (camera) {
      const data = await camera.recordAsync();
      setRecording(data.uri);
      console.log('Recording URI:', data.uri);
    }
  };

  // Stop video recording
  const stopVideo = () => {
    if (camera) {
      camera.stopRecording();
    }
  };

  if (hasCameraPermission === null || hasAudioPermission === null) {
    return <View />;
  }
  if (hasCameraPermission === false || hasAudioPermission === false) {
    return <Text>No access to camera or audio</Text>;
  }

  return (
    <View style={styles.container}>
      <View style={styles.cameraContainer}>
        {type && (
          <Camera
            ref={(ref) => setCamera(ref)}
            style={styles.fixedRatio}
            type={type}
            ratio={'4:3'}
            flashMode={flashMode}
            autoFocus={autoFocus}
          />
        )}
      </View>

      {/* Flip Camera Button */}
      <Button
        title="Flip Camera"
        onPress={() => {
          setType(
            type === Camera.Constants.Type.back
              ? Camera.Constants.Type.front
              : Camera.Constants.Type.back
          );
        }}
      />

      {/* Toggle Flash */}
      <Button
        title={`Flash: ${flashMode === Camera.Constants.FlashMode.off ? 'Off' : 'On'}`}
        onPress={() =>
          setFlashMode(
            flashMode === Camera.Constants.FlashMode.off
              ? Camera.Constants.FlashMode.on
              : Camera.Constants.FlashMode.off
          )
        }
      />

      {/* Toggle AutoFocus */}
      <Button
        title={`AutoFocus: ${autoFocus === Camera.Constants.AutoFocus.on ? 'On' : 'Off'}`}
        onPress={() =>
          setAutoFocus(
            autoFocus === Camera.Constants.AutoFocus.on
              ? Camera.Constants.AutoFocus.off
              : Camera.Constants.AutoFocus.on
          )
        }
      />

      {/* Start and Stop Recording */}
      <Button title="Start Recording" onPress={takeVideo} />
      <Button title="Stop Recording" onPress={stopVideo} />

      {/* Video Playback */}
      {recording && (
        <>
          <Video
            ref={video}
            style={styles.video}
            source={{ uri: recording }}
            useNativeControls
            resizeMode="contain"
            isLooping
            onPlaybackStatusUpdate={(status) => setStatus(() => status)}
          />
          <Button
            title={status.isPlaying ? 'Pause' : 'Play'}
            onPress={() =>
              status.isPlaying
                ? video.current.pauseAsync()
                : video.current.playAsync()
            }
          />
        </>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  cameraContainer: {
    flex: 1,
    flexDirection: 'row',
  },
  fixedRatio: {
    flex: 1,
    aspectRatio: 3 / 4,
  },
  video: {
    flex: 1,
    width: '100%',
  },
});
