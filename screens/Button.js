import React from 'react';
import { TouchableOpacity, Text, StyleSheet } from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';

const Button = ({ title, onPress, icon, color = '#fff' }) => {
  return (
    <TouchableOpacity onPress={onPress} style={styles.button}>
      <MaterialIcons name={icon} size={24} color={color} />
      <Text style={styles.text}>{title}</Text>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  button: {
    backgroundColor: '#1D1C1F',
    padding: 10,
    marginVertical: 10,
    borderRadius: 10,
    flexDirection: 'row',
    alignItems: 'center',
  },
  text: {
    color: '#fff',
    fontSize: 16,
    marginLeft: 10,
  },
});

export default Button;
