import React, { useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import Svg, { Polyline } from 'react-native-svg';

const LoadingAlphaOne = () => {
  const navigation = useNavigation();

  useEffect(() => {
    const timer = setTimeout(() => {
      navigation.navigate('SkeletonAlpha'); // Navigate after 3 seconds
    }, 3000);
    return () => clearTimeout(timer);
  }, [navigation]);

  return (
    <View style={styles.container}>
      <View style={styles.loading}>
        <Svg width="64px" height="48px">
          <Polyline
            points="0.157 23.954, 14 23.954, 21.843 48, 43 0, 50 24, 64 24"
            id="back"
            style={styles.back}
          />
          <Polyline
            points="0.157 23.954, 14 23.954, 21.843 48, 43 0, 50 24, 64 24"
            id="front"
            style={styles.front}
          />
        </Svg>
      </View>
      <Text style={styles.text}>Analyzing Video</Text>
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
  loading: {
    marginBottom: 20,
  },
  back: {
    fill: 'none',
    stroke: '#F4F6F8',
    strokeWidth: 3,
    strokeLinecap: 'round',
    strokeLinejoin: 'round',
  },
  front: {
    fill: 'none',
    stroke: '#F34533',
    strokeWidth: 3,
    strokeLinecap: 'round',
    strokeLinejoin: 'round',
    strokeDasharray: '48, 144',
    strokeDashoffset: 192,
    animationKeyframes: [
      { '72.5%': { opacity: 0 } },
      { to: { strokeDashoffset: 0 } },
    ],
    animation: 'dash 1.4s linear infinite',
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

export default LoadingAlphaOne;
