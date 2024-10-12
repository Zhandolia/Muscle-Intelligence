// src/screens/PostureAnalysisScreen.js
import React, { useState } from 'react';
import { View, Text, Button, StyleSheet } from 'react-native';
// import { RNCamera } from 'react-native-camera';

const PostureAnalysisScreen = ({ navigation }) => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const startAnalysis = () => {
    setIsAnalyzing(true);
    // Placeholder for actual posture analysis logic
  };

  const stopAnalysis = () => {
    setIsAnalyzing(false);
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Posture Analysis</Text>
      {isAnalyzing ? (
        <View style={styles.cameraView}>
          <RNCamera style={styles.camera} type={RNCamera.Constants.Type.back} />
          <Button title="Stop Analysis" onPress={stopAnalysis} />
        </View>
      ) : (
        <Button title="Start Posture Analysis" onPress={startAnalysis} />
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f0f0f0',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  cameraView: {
    flex: 1,
    width: '100%',
    alignItems: 'center',
    justifyContent: 'center',
  },
  camera: {
    width: '100%',
    height: '60%',
    marginBottom: 20,
  },
});

export default PostureAnalysisScreen;
