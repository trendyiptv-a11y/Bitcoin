import React, { useState } from 'react';
import { View, Text, TouchableOpacity, ScrollView, Switch, Alert } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { styles } from '@/styles/settings.styles';

export default function Settings() {
  const [notifications, setNotifications] = useState(true);
  const [darkMode, setDarkMode] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const handleClearCache = () => {
    Alert.alert('Cache șters', 'Datele cache au fost șterse cu succes.');
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Preferințe</Text>

        <LinearGradient colors={['#1a1a2e', '#16213e']} style={styles.settingItem}>
          <View style={styles.settingContent}>
            <Text style={styles.settingLabel}>Notificări</Text>
            <Text style={styles.settingDescription}>Primește alertele pentru schimbări majore</Text>
          </View>
          <Switch
            value={notifications}
            onValueChange={setNotifications}
            trackColor={{ false: '#333', true: '#4CAF50' }}
            thumbColor={notifications ? '#4CAF50' : '#f4f3f4'}
          />
        </LinearGradient>

        <LinearGradient colors={['#1a1a2e', '#16213e']} style={styles.settingItem}>
          <View style={styles.settingContent}>
            <Text style={styles.settingLabel}>Mod Întunecat</Text>
            <Text style={styles.settingDescription}>Afișare cu fundal închis</Text>
          </View>
          <Switch
            value={darkMode}
            onValueChange={setDarkMode}
            trackColor={{ false: '#333', true: '#00d4ff' }}
            thumbColor={darkMode ? '#00d4ff' : '#f4f3f4'}
          />
        </LinearGradient>

        <LinearGradient colors={['#1a1a2e', '#16213e']} style={styles.settingItem}>
          <View style={styles.settingContent}>
            <Text style={styles.settingLabel}>Reîncărcare Automată</Text>
            <Text style={styles.settingDescription}>Actualizează datele la 5 minute</Text>
          </View>
          <Switch
            value={autoRefresh}
            onValueChange={setAutoRefresh}
            trackColor={{ false: '#333', true: '#FF9800' }}
            thumbColor={autoRefresh ? '#FF9800' : '#f4f3f4'}
          />
        </LinearGradient>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Date și Cache</Text>

        <TouchableOpacity onPress={handleClearCache}>
          <LinearGradient colors={['#2d1a1a', '#4d2d2d']} style={styles.button}>
            <Text style={styles.buttonText}>Șterge Cache</Text>
          </LinearGradient>
        </TouchableOpacity>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Despre</Text>

        <LinearGradient colors={['#1a1a2e', '#16213e']} style={styles.infoCard}>
          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>Versiune Aplicație</Text>
            <Text style={styles.infoValue}>1.0.0</Text>
          </View>
          <View style={[styles.infoRow, styles.borderTop]}>
            <Text style={styles.infoLabel}>Server API</Text>
            <Text style={styles.infoValue}>CohesivX BTC</Text>
          </View>
          <View style={[styles.infoRow, styles.borderTop]}>
            <Text style={styles.infoLabel}>Status</Text>
            <View style={styles.statusBadge}>
              <Text style={styles.statusText}>Online</Text>
            </View>
          </View>
        </LinearGradient>
      </View>

      <View style={styles.footer} />
    </ScrollView>
  );
}
