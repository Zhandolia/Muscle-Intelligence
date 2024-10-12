// VideoUploadScreen.js
import React, { useState } from 'react';
import { View, Text, Button, Alert } from 'react-native';

const VideoUploadScreen = () => {
  const [selectedWorkout, setSelectedWorkout] = useState(null);

  const handleFinalize = () => {
    if (selectedWorkout) {
      alert(`Workout selected: ${selectedWorkout}`);
      // Proceed with upload or other actions
    } else {
      alert('Please select a workout.');
    }
  };

  return (
    <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
      <Text>Select a workout type:</Text>
      <Button title="Yoga" onPress={() => setSelectedWorkout('Yoga')} />
      <Button title="Cardio" onPress={() => setSelectedWorkout('Cardio')} />
      <Button title="Strength" onPress={() => setSelectedWorkout('Strength')} />

      <Button title="Finalize" onPress={handleFinalize} />
    </View>
  );
};

export default VideoUploadScreen;
