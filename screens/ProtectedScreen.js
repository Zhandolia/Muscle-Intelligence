import React from 'react';
import { View, Text } from 'react-native';
import useProtectedRoute from '../hooks/useProtectedRoute'; // Adjust the path based on your structure

const ProtectedScreen = ({ session }) => {
  useProtectedRoute(session); // Apply the protected route logic

  return (
    <View>
      <Text>Protected Content</Text>
    </View>
  );
};

export default ProtectedScreen;
