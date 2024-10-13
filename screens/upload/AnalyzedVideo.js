import React from 'react';
import { View, Text, Image, StyleSheet, TouchableOpacity } from 'react-native';
import Svg, { Polyline } from 'react-native-svg';
import { useNavigation } from '@react-navigation/native';

const AnalyzedVideo = () => {
    const navigation = useNavigation();
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Confirm Skeleton</Text>
      <Svg width="64px" height="48px">
        <Image source={require('../../assets/output1.gif')} style={styles.video} />
      </Svg>
      <TouchableOpacity>
        <Text style={[styles.text, styles.orange]}>Didn’t match? Regenerate here!</Text>
      </TouchableOpacity>
      <TouchableOpacity style={styles.button}>
        <Text style={styles.buttonText}>Confirm →</Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F9FAFB',
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1D1C1F',
    marginBottom: 20,
  },
  video: {
    width: 300,
    height: 200,
    marginBottom: 20,
  },
  text: {
    fontSize: 16,
    marginBottom: 5,
  },
  orange: {
    color: '#F34533',
  },
  button: {
    marginTop: 20,
    backgroundColor: '#F34533',
    padding: 10,
    borderRadius: 5,
  },
  buttonText: {
    color: '#F9FAFB',
    fontSize: 16,
  },
});

export default AnalyzedVideo;
