import React, { useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import Svg, { Polyline } from 'react-native-svg';

const AnalyzingVideo = () => {
    const navigation = useNavigation();
    
    return (
      <View style={styles.container}>
        <View className="loading">
          <Svg width="64px" height="48px">
            <Polyline
              points="0.157 23.954, 14 23.954, 21.843 48, 43 0, 50 24, 64 24"
              id="back"
            />
            <Polyline
              points="0.157 23.954, 14 23.954, 21.843 48, 43 0, 50 24, 64 24"
              id="front"
            />
          </Svg>
        </View>
        <Text style={styles.text}>Analyzing Video</Text>
        <TouchableOpacity style={styles.button} onPress={() => navigation.navigate('AnalyzedVideo')}>
          <Text style={styles.buttonText}>Next →</Text>
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
  text: {
    marginTop: 20,
    fontSize: 18,
    color: '#1D1C1F',
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

export default AnalyzingVideo;
